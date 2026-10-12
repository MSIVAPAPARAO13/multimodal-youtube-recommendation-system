"""
Explicit and Restartable Embedding Backfill Script.

Experiment / Phase 9 Mode: TEXT_COENGAGEMENT_V2
Reads video titles from database or MicroLens dataset, generates 512-D
normalized representations using the verified Phase 8C checkpoint, and
persists embeddings idempotently.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sqlalchemy.orm import Session
from transformers import BertModel, BertTokenizer

from src.db.config import get_settings
from src.db.database import get_database
from src.db.models import Video
from src.db.repository import VideoRepository
from src.models.text_projection_model import load_phase_8c_projection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_vector_index")

DEFAULT_CHECKPOINT = "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt"
MICROLENS_TITLES_CSV = "data/raw/microlens/MicroLens-50k_titles.csv"


def load_encoder(checkpoint_path: str, device: torch.device):
    """Load tokenizer, frozen BERT model, and verified Phase 8C projection head."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase 8C checkpoint missing at {checkpoint_path}. Run Phase 8C first.")

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    logger.info("Loading Phase 8C checkpoint: %s (SHA256: %s)", ckpt_path, sha256)

    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True).to(device)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    for p in bert.parameters():
        p.requires_grad = False
    bert.eval()

    proj = load_phase_8c_projection(checkpoint_path=str(ckpt_path), device=device)
    return tokenizer, bert, proj, sha256


def batch_encode_titles(
    titles: List[str],
    tokenizer: BertTokenizer,
    bert: BertModel,
    proj: torch.nn.Module,
    device: torch.device,
) -> List[List[float]]:
    """Generate 512-D normalized embeddings for a batch of titles."""
    tokens = tokenizer(
        titles,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        out = bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = out.last_hidden_state[:, 0, :]
        embs = proj(cls_feat)

    return embs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="Backfill vector embeddings using Phase 8C checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Maximum items to backfill")
    parser.add_argument("--from-csv", type=str, default=None, help="Optional path to titles CSV")
    parser.add_argument("--from-microlens-catalog", action="store_true", default=False, help="Index the 2960 items from MicroLens evaluation catalog")
    parser.add_argument("--catalog-users", type=int, default=1000, help="Users to define MicroLens catalog")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip items already embedded")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Generate embeddings without persisting")
    args = parser.parse_args()

    device = torch.device("cpu")
    tokenizer, bert, proj, ckpt_sha = load_encoder(args.checkpoint, device)

    from src.services.vector_index_service import VectorIndexService
    index_service = VectorIndexService()

    db = get_database()
    session = db.get_session_direct()
    repo = VideoRepository(session)

    items_to_process: List[Tuple[str, str, Optional[torch.Tensor]]] = []  # (video_id, title, cached_raw_feature)

    if args.from_microlens_catalog:
        from src.data.text_coengagement import TextCoengagementBuilder, BertFeatureExtractor
        logger.info("Loading MicroLens evaluation catalog items (catalog_users=%d)...", args.catalog_users)
        builder = TextCoengagementBuilder(seed=42)
        ds = builder.build_dataset(max_users=args.catalog_users)
        unique_items = ds["unique_items"]
        titles_map = ds["titles_map"]

        cache_path = Path("data/cache") / f"microlens_bert_{len(unique_items)}.pt"
        extractor = BertFeatureExtractor(device=device)
        raw_features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

        for iid in unique_items:
            vid = str(iid)
            title = titles_map.get(iid, f"Video {vid}")
            raw_feat = raw_features.get(iid)
            items_to_process.append((vid, title, raw_feat))
            if args.limit and len(items_to_process) >= args.limit:
                break
    elif args.from_csv:
        csv_path = Path(args.from_csv)
        logger.info("Reading video items from CSV: %s", csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("item") or row.get("item_id") or row.get("video_id") or "").strip()
                title = str(row.get("title", "")).strip()
                if vid and title:
                    items_to_process.append((vid, title, None))
                    if args.limit and len(items_to_process) >= args.limit:
                        break
    else:
        # Read from database
        logger.info("Reading videos from database...")
        all_videos = repo.get_all(limit=args.limit or 50000)
        for v in all_videos:
            if args.skip_existing and v.embedding is not None:
                continue
            if v.title and v.title.strip():
                items_to_process.append((v.video_id, v.title.strip(), None))

    logger.info("Total items queued for backfill: %d (dry_run=%s)", len(items_to_process), args.dry_run)

    if not items_to_process:
        logger.info("No items requiring backfill.")
        session.close()
        return

    processed = 0
    inserted = 0
    updated = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    # Pre-fetch existing video IDs if skip_existing
    existing_videos = {}
    if args.skip_existing:
        batch_ids_all = [item[0] for item in items_to_process]
        for chunk_start in range(0, len(batch_ids_all), 1000):
            chunk_ids = batch_ids_all[chunk_start:chunk_start + 1000]
            existing_videos.update(repo.get_by_video_ids(chunk_ids))

    for i in range(0, len(items_to_process), args.batch_size):
        batch = items_to_process[i:i + args.batch_size]
        batch_ids = [item[0] for item in batch]
        batch_titles = [item[1] for item in batch]
        batch_feats = [item[2] for item in batch]

        try:
            if all(f is not None for f in batch_feats):
                # Fast path: project cached 768-D features directly
                feat_tensor = torch.stack(batch_feats).to(device)
                with torch.no_grad():
                    embs = proj(feat_tensor).cpu().tolist()
            else:
                embs = batch_encode_titles(batch_titles, tokenizer, bert, proj, device)

            if not args.dry_run:
                for vid, title, emb in zip(batch_ids, batch_titles, embs):
                    valid_emb = index_service.validate_embedding(emb)
                    ex_vid = existing_videos.get(vid)
                    if args.skip_existing and ex_vid and ex_vid.embedding is not None:
                        skipped += 1
                        continue
                    if ex_vid:
                        updated += 1
                    else:
                        inserted += 1
                    repo.upsert_embedding(video_id=vid, embedding=valid_emb, title=title)
            else:
                for emb in embs:
                    index_service.validate_embedding(emb)

            processed += len(batch)
            logger.info("Processed %d / %d items (%.1f%%)", processed, len(items_to_process), (processed / len(items_to_process)) * 100.0)
        except Exception as e:
            session.rollback()
            logger.error("Error processing batch starting at index %d: %s", i, e)
            failed += len(batch)

    dur = time.time() - start_time
    logger.info(
        "Backfill finished in %.2fs: processed=%d, inserted=%d, updated=%d, skipped=%d, failed=%d",
        dur, processed, inserted, updated, skipped, failed
    )
    session.close()


if __name__ == "__main__":
    main()
