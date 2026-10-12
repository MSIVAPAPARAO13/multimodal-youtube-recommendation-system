"""
Semantic Text Retrieval Evaluation Script (TASK 14).

Evaluates whether semantically related video documents receive higher cosine similarity
than unrelated documents using the structured text modality pipeline.

Metrics reported:
- Precision@K (K=1, K=2)
- Recall@K (K=1, K=2)
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)

DISCLAIMER:
This evaluation is executed on a controlled/synthetic fixture of curated video metadata.
It measures representation clustering and semantic ranking of the text modality pipeline,
not real-world user engagement or live recommendation quality.
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn.functional as F

from src.features.text_processor import YouTubeTextInput, TextProcessor
from src.features.text_encoder import TextEncoder
from src.services.multimodal_inference import TextEmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_fixtures(fixture_path: Path) -> list[dict[str, Any]]:
    """Load evaluation fixture JSON."""
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found at {fixture_path}")
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    fixture_path: Optional[Path] = None,
    pretrained: bool = True,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Run retrieval evaluation across synthetic video records.
    """
    if fixture_path is None:
        fixture_path = root_dir / "data" / "fixtures" / "synthetic_eval_videos.json"

    records = load_fixtures(fixture_path)
    n_docs = len(records)
    print(f"Loaded {n_docs} synthetic evaluation records from {fixture_path.name}")

    # Initialize TextEmbeddingService with TextEncoder
    torch_device = torch.device(device)
    print(f"Initializing BERT TextEncoder (pretrained={pretrained}, device={device})...")
    t0 = time.perf_counter()
    encoder = TextEncoder(
        model_name="bert-base-uncased",
        feature_dim=512,
        pretrained=pretrained,
        freeze_backbone=True
    ).to(torch_device)
    encoder.eval()

    text_service = TextEmbeddingService(
        text_encoder=encoder,
        device=torch_device,
        feature_dim=512
    )
    init_time_ms = (time.perf_counter() - t0) * 1000
    print(f"TextEncoder ready ({init_time_ms:.1f} ms)")

    # Compute embeddings for all records
    embeddings_list = []
    metadata_list = []

    print("\nEncoding documents with structured section formatting and chunking...")
    for rec in records:
        text_input = YouTubeTextInput(
            title=rec["title"],
            description=rec.get("description"),
            tags=rec.get("tags"),
            transcript=rec.get("transcript")
        )
        res = text_service.encode(text_input)
        emb = res.embedding  # (1, 512)
        # L2-normalize so dot products equal cosine similarities
        emb_norm = F.normalize(emb, p=2, dim=-1)
        embeddings_list.append(emb_norm)
        metadata_list.append({
            "video_id": rec["video_id"],
            "category": rec["category"],
            "title": rec["title"],
            "token_count": res.token_count,
            "chunk_count": res.chunk_count,
        })

    # Stack into tensor of shape (N, 512)
    doc_vectors = torch.cat(embeddings_list, dim=0)  # (N, 512)

    # Compute full pairwise similarity matrix (N, N)
    sim_matrix = torch.matmul(doc_vectors, doc_vectors.T).cpu().numpy()

    # Metrics accumulators
    p_at_1_list = []
    p_at_2_list = []
    r_at_1_list = []
    r_at_2_list = []
    mrr_list = []
    ap_list = []

    categories = set(rec["category"] for rec in records)
    category_results: dict[str, list[float]] = {cat: [] for cat in categories}

    for i in range(n_docs):
        query_cat = metadata_list[i]["category"]
        # Relevant documents: same category, excluding self
        relevant_indices = set(
            j for j in range(n_docs) if j != i and metadata_list[j]["category"] == query_cat
        )
        total_relevant = len(relevant_indices)
        if total_relevant == 0:
            continue

        # Sort all other documents by similarity score descending
        ranked_candidates = [
            (j, sim_matrix[i, j]) for j in range(n_docs) if j != i
        ]
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        ranked_indices = [c[0] for c in ranked_candidates]

        # Precision@1 & Recall@1
        top_1 = ranked_indices[:1]
        rel_in_top_1 = sum(1 for idx in top_1 if idx in relevant_indices)
        p_at_1 = rel_in_top_1 / 1.0
        r_at_1 = rel_in_top_1 / float(total_relevant)
        p_at_1_list.append(p_at_1)
        r_at_1_list.append(r_at_1)

        # Precision@2 & Recall@2
        top_2 = ranked_indices[:2]
        rel_in_top_2 = sum(1 for idx in top_2 if idx in relevant_indices)
        p_at_2 = rel_in_top_2 / 2.0
        r_at_2 = rel_in_top_2 / float(total_relevant)
        p_at_2_list.append(p_at_2)
        r_at_2_list.append(r_at_2)

        # Reciprocal Rank (MRR)
        first_rel_rank = None
        for rank_pos, idx in enumerate(ranked_indices, start=1):
            if idx in relevant_indices:
                first_rel_rank = rank_pos
                break
        rr = (1.0 / first_rel_rank) if first_rel_rank else 0.0
        mrr_list.append(rr)

        # Average Precision (AP)
        num_hits = 0
        sum_precisions = 0.0
        for rank_pos, idx in enumerate(ranked_indices, start=1):
            if idx in relevant_indices:
                num_hits += 1
                sum_precisions += num_hits / rank_pos
        ap = (sum_precisions / float(total_relevant)) if total_relevant > 0 else 0.0
        ap_list.append(ap)

        category_results[query_cat].append(p_at_1)

    mean_p1 = sum(p_at_1_list) / len(p_at_1_list)
    mean_p2 = sum(p_at_2_list) / len(p_at_2_list)
    mean_r1 = sum(r_at_1_list) / len(r_at_1_list)
    mean_r2 = sum(r_at_2_list) / len(r_at_2_list)
    mean_mrr = sum(mrr_list) / len(mrr_list)
    mean_map = sum(ap_list) / len(ap_list)

    print("\n" + "=" * 65)
    print("  CONTROLLED TEXT RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Total Evaluated Documents:  {n_docs}")
    print(f"  Distinct Semantic Groups:   {len(categories)}")
    print(f"  Relevant Items Per Query:   2 (out of 14 candidates)")
    print("-" * 65)
    print(f"  Precision@1:                {mean_p1 * 100:6.2f}%")
    print(f"  Precision@2:                {mean_p2 * 100:6.2f}%")
    print(f"  Recall@1:                   {mean_r1 * 100:6.2f}%")
    print(f"  Recall@2:                   {mean_r2 * 100:6.2f}%")
    print(f"  MRR (Mean Reciprocal Rank): {mean_mrr:8.4f}")
    print(f"  MAP (Mean Average Prec.):   {mean_map:8.4f}")
    print("-" * 65)
    print("  Category-Level Top-1 Accuracy:")
    for cat, scores in sorted(category_results.items()):
        acc = (sum(scores) / len(scores)) * 100 if scores else 0.0
        print(f"    - {cat:22s}: {acc:5.1f}%")
    print("=" * 65)
    print("  Note: Synthetic/controlled baseline test. Measures representation")
    print("  clustering ability on structured text, not live recommendation quality.")
    print("=" * 65)

    return {
        "num_documents": n_docs,
        "num_categories": len(categories),
        "precision_at_1": mean_p1,
        "precision_at_2": mean_p2,
        "recall_at_1": mean_r1,
        "recall_at_2": mean_r2,
        "mrr": mean_mrr,
        "map": mean_map,
        "category_accuracy": {
            cat: (sum(scores) / len(scores)) if scores else 0.0
            for cat, scores in category_results.items()
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate text retrieval baseline")
    parser.add_argument("--uninitialized", action="store_true", help="Use uninitialized BERT weights (offline/fast test)")
    parser.add_argument("--device", default="cpu", help="Device to run inference on ('cpu' or 'cuda')")
    args = parser.parse_args()

    evaluate_retrieval(
        pretrained=not args.uninitialized,
        device=args.device
    )
