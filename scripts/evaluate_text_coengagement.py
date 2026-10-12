"""
Evaluation Protocol for Text-Only Metadata Co-Engagement Experiment.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V1
Evaluates recommendation retrieval performance on held-out test users:
- Untrained Projection Baseline vs. Trained Projection Model vs. Raw Frozen BERT.
- Computes Recall@K (K=1, 5, 10), Precision@K, MRR, and NDCG@10.
- Excludes self-matches from ranking to prevent trivial hits.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.text_coengagement import (
    TextCoengagementBuilder,
    BertFeatureExtractor,
)
from src.models.text_projection_model import TextProjectionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_text_coengagement")

EXPERIMENT_ID = "EXP_MICROLENS_TEXT_COENGAGEMENT_V1"


def compute_metrics(
    query_embeddings: torch.Tensor,
    query_item_ids: List[int],
    candidate_embeddings: torch.Tensor,
    candidate_item_ids: List[int],
    ground_truth_map: Dict[int, Set[int]],
    top_k_list: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval metrics against candidate gallery with self-match exclusion.
    """
    q_norm = F.normalize(query_embeddings, p=2, dim=-1)
    c_norm = F.normalize(candidate_embeddings, p=2, dim=-1)

    sim_matrix = torch.matmul(q_norm, c_norm.T).cpu().numpy()

    recalls = {f"Recall@{k}": [] for k in top_k_list}
    precisions = {f"Precision@{k}": [] for k in top_k_list}
    ndcgs = {f"NDCG@{k}": [] for k in top_k_list}
    mrrs = []

    valid_queries = 0

    for i, q_id in enumerate(query_item_ids):
        gt = ground_truth_map.get(q_id, set())
        if not gt:
            continue

        valid_queries += 1
        # Sort candidate indices by similarity descending
        ranked_indices = np.argsort(-sim_matrix[i])

        # Exclude self-match from candidate ranking
        ranked_cand_ids = [
            candidate_item_ids[idx] for idx in ranked_indices
            if candidate_item_ids[idx] != q_id
        ]

        # MRR
        rr = 0.0
        for rank, item in enumerate(ranked_cand_ids, start=1):
            if item in gt:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # Recall@K, Precision@K, NDCG@K
        for k in top_k_list:
            top_k_items = ranked_cand_ids[:k]
            hits = sum(1 for item in top_k_items if item in gt)

            recalls[f"Recall@{k}"].append(hits / float(len(gt)))
            precisions[f"Precision@{k}"].append(hits / float(k))

            # NDCG@K
            dcg = 0.0
            for rank, item in enumerate(top_k_items, start=1):
                if item in gt:
                    dcg += 1.0 / math.log2(rank + 1)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gt), k) + 1))
            ndcgs[f"NDCG@{k}"].append(dcg / idcg if idcg > 0.0 else 0.0)

    if valid_queries == 0:
        return {"valid_queries": 0}

    results = {
        "valid_queries": valid_queries,
        "MRR": float(np.mean(mrrs)),
    }
    for k in top_k_list:
        results[f"Recall@{k}"] = float(np.mean(recalls[f"Recall@{k}"]))
        results[f"Precision@{k}"] = float(np.mean(precisions[f"Precision@{k}"]))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs[f"NDCG@{k}"]))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate text co-engagement model on test split")
    parser.add_argument("--checkpoint-path", type=str, default=f"data/checkpoints/{EXPERIMENT_ID}/best.pt")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output-json", type=str, default=f"data/checkpoints/{EXPERIMENT_ID}/evaluation_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = Path(args.checkpoint_path)

    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s. Please run scripts/train_text_coengagement.py first.", ckpt_path)
        return 1

    # 1. Rebuild Dataset with same seed
    logger.info("Loading MicroLens test split (max_users=%s, seed=%d)...", args.max_users, args.seed)
    builder = TextCoengagementBuilder(seed=args.seed)
    dataset_dict = builder.build_dataset(max_users=args.max_users)
    test_pairs = dataset_dict["test_pairs"]
    stats = dataset_dict["stats"]
    unique_items = dataset_dict["unique_items"]

    logger.info("Test split loaded: %d test pairs across %d test users.", len(test_pairs), stats["test_users"])

    # 2. Load Cached BERT Features
    cache_path = Path(args.cache_dir) / f"microlens_bert_{len(unique_items)}.pt"
    extractor = BertFeatureExtractor(device=device)
    features = extractor.extract_features(
        item_ids=unique_items,
        titles_map=dataset_dict["titles_map"],
        cache_file=cache_path,
    )

    # 3. Ground truth mapping: anchor_item -> set of next positive items in test
    ground_truth_map: Dict[int, Set[int]] = {}
    for p in test_pairs:
        ground_truth_map.setdefault(p.anchor_item, set()).add(p.positive_item)

    query_item_ids = sorted(list(ground_truth_map.keys()))
    candidate_item_ids = list(unique_items)

    # Candidate gallery features: (N_cand, 768)
    cand_bert_tensor = torch.stack([features[item_id] for item_id in candidate_item_ids], dim=0)
    query_bert_tensor = torch.stack([features[q_id] for q_id in query_item_ids], dim=0)

    # 4. Evaluation 1: Raw Frozen BERT (768-D baseline)
    logger.info("Evaluating Baseline 1: Raw Pretrained BERT (768-D)...")
    bert_metrics = compute_metrics(
        query_embeddings=query_bert_tensor,
        query_item_ids=query_item_ids,
        candidate_embeddings=cand_bert_tensor,
        candidate_item_ids=candidate_item_ids,
        ground_truth_map=ground_truth_map,
    )
    logger.info("Raw BERT Baseline Metrics: %s", bert_metrics)

    # 5. Evaluation 2: Untrained Projection Baseline (Randomly initialized projection head)
    torch.manual_seed(args.seed)
    untrained_model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512)
    untrained_model.eval()

    with torch.no_grad():
        untrained_query_emb = untrained_model(query_bert_tensor)
        untrained_cand_emb = untrained_model(cand_bert_tensor)

    logger.info("Evaluating Baseline 2: Untrained Projection Head (Random 512-D)...")
    untrained_metrics = compute_metrics(
        query_embeddings=untrained_query_emb,
        query_item_ids=query_item_ids,
        candidate_embeddings=untrained_cand_emb,
        candidate_item_ids=candidate_item_ids,
        ground_truth_map=ground_truth_map,
    )
    logger.info("Untrained Projection Metrics: %s", untrained_metrics)

    # 6. Evaluation 3: Trained Co-Engagement Projection Model
    trained_model = TextProjectionModel(input_dim=768, hidden_dim=768, output_dim=512)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trained_model.load_state_dict(ckpt["model_state_dict"])
    trained_model.eval()

    with torch.no_grad():
        trained_query_emb = trained_model(query_bert_tensor)
        trained_cand_emb = trained_model(cand_bert_tensor)

    logger.info("Evaluating Trained Co-Engagement Model (%s)...", ckpt_path)
    trained_metrics = compute_metrics(
        query_embeddings=trained_query_emb,
        query_item_ids=query_item_ids,
        candidate_embeddings=trained_cand_emb,
        candidate_item_ids=candidate_item_ids,
        ground_truth_map=ground_truth_map,
    )
    logger.info("Trained Co-Engagement Metrics: %s", trained_metrics)

    # Compute Relative Improvements
    rel_improvements = {}
    for k in ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"]:
        base_val = untrained_metrics.get(k, 0.0)
        train_val = trained_metrics.get(k, 0.0)
        if base_val > 0.0:
            rel_improvements[k] = round(((train_val - base_val) / base_val) * 100.0, 2)
        else:
            rel_improvements[k] = None

    eval_summary = {
        "experiment_id": EXPERIMENT_ID,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
        "test_users": stats["test_users"],
        "test_pairs": len(test_pairs),
        "unique_queries": len(query_item_ids),
        "candidate_gallery_size": len(candidate_item_ids),
        "metrics": {
            "raw_bert_768d": bert_metrics,
            "untrained_projection_baseline": untrained_metrics,
            "trained_coengagement_model": trained_metrics,
        },
        "relative_improvements_pct_over_untrained": rel_improvements,
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Evaluation results written to %s", out_file)
    print("\n" + "="*60)
    print("PHASE 8B EVALUATION COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<12} | {'Raw BERT':<12} | {'Untrained':<12} | {'Trained Model':<14} | {'Improvement'}")
    print("-"*60)
    for k in ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"]:
        b_raw = bert_metrics.get(k, 0.0)
        b_un = untrained_metrics.get(k, 0.0)
        t_mod = trained_metrics.get(k, 0.0)
        imp = rel_improvements.get(k)
        imp_str = f"+{imp:.1f}%" if imp is not None and imp > 0 else (f"{imp:.1f}%" if imp is not None else "N/A")
        print(f"{k:<12} | {b_raw:.4f}       | {b_un:.4f}       | {t_mod:.4f}         | {imp_str}")
    print("="*60)

    return 0


if __name__ == "__main__":
    main()
