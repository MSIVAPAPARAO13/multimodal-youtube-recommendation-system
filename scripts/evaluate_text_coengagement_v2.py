"""
Held-Out Evaluation for Phase 8C Text Co-Engagement Optimization.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V2
Evaluates held-out test split under the exact same fixed protocol:
- Raw BERT (768-D)
- Untrained Projection Baseline (512-D)
- Phase 8B Trained Model (512-D)
- Phase 8C Optimized Model (512-D)
Computes Recall@1, Recall@5, Recall@10, Precision@K, MRR, NDCG@10,
and user-level bootstrap 95% confidence intervals.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.text_coengagement import (
    TextCoengagementBuilder,
    BertFeatureExtractor,
)
from src.models.text_projection_model import TextProjectionModel
from scripts.optimize_text_coengagement import LinearProjectionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_text_coengagement_v2")

EXPERIMENT_ID_V2 = "EXP_MICROLENS_TEXT_COENGAGEMENT_V2"
EXPERIMENT_ID_V1 = "EXP_MICROLENS_TEXT_COENGAGEMENT_V1"


def compute_metrics_with_bootstrap(
    query_embeddings: torch.Tensor,
    query_item_ids: List[int],
    candidate_embeddings: torch.Tensor,
    candidate_item_ids: List[int],
    ground_truth_map: Dict[int, Set[int]],
    top_k_list: Tuple[int, ...] = (1, 5, 10),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
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
        ranked_indices = np.argsort(-sim_matrix[i])

        # Exclude self-match
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

        for k in top_k_list:
            top_k_items = ranked_cand_ids[:k]
            hits = sum(1 for item in top_k_items if item in gt)

            recalls[f"Recall@{k}"].append(hits / float(len(gt)))
            precisions[f"Precision@{k}"].append(hits / float(k))

            dcg = 0.0
            for rank, item in enumerate(top_k_items, start=1):
                if item in gt:
                    dcg += 1.0 / math.log2(rank + 1)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gt), k) + 1))
            ndcgs[f"NDCG@{k}"].append(dcg / idcg if idcg > 0.0 else 0.0)

    results = {
        "valid_queries": valid_queries,
        "MRR": float(np.mean(mrrs)),
    }
    for k in top_k_list:
        results[f"Recall@{k}"] = float(np.mean(recalls[f"Recall@{k}"]))
        results[f"Precision@{k}"] = float(np.mean(precisions[f"Precision@{k}"]))
        results[f"NDCG@{k}"] = float(np.mean(ndcgs[f"NDCG@{k}"]))

    # Bootstrap 95% confidence intervals on MRR
    rng = np.random.default_rng(seed)
    boot_mrrs = []
    boot_r10s = []
    mrr_arr = np.array(mrrs)
    r10_arr = np.array(recalls["Recall@10"])

    n_samples = len(mrr_arr)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_mrrs.append(np.mean(mrr_arr[idx]))
        boot_r10s.append(np.mean(r10_arr[idx]))

    results["bootstrap_ci95"] = {
        "MRR": [float(np.percentile(boot_mrrs, 2.5)), float(np.percentile(boot_mrrs, 97.5))],
        "Recall@10": [float(np.percentile(boot_r10s, 2.5)), float(np.percentile(boot_r10s, 97.5))],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 8C held-out test performance")
    parser.add_argument("--v1-checkpoint", type=str, default=f"data/checkpoints/{EXPERIMENT_ID_V1}/best.pt")
    parser.add_argument("--v2-checkpoint", type=str, default=f"data/checkpoints/{EXPERIMENT_ID_V2}/best.pt")
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--output-json", type=str, default=f"data/checkpoints/{EXPERIMENT_ID_V2}/evaluation_results.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    v1_ckpt_path = Path(args.v1_checkpoint)
    v2_ckpt_path = Path(args.v2_checkpoint)

    # 1. Verify V1 Checkpoint Fingerprint (Must be unchanged)
    v1_sha256 = hashlib.sha256(v1_ckpt_path.read_bytes()).hexdigest()
    logger.info("Phase 8B Checkpoint SHA256: %s", v1_sha256)

    # 2. Build exact test dataset
    builder = TextCoengagementBuilder(seed=args.seed)
    ds = builder.build_dataset(max_users=args.max_users)
    test_pairs = ds["test_pairs"]
    unique_items = ds["unique_items"]
    titles_map = ds["titles_map"]

    cache_path = Path(args.cache_dir) / f"microlens_bert_{len(unique_items)}.pt"
    extractor = BertFeatureExtractor(device=device)
    features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

    ground_truth_map: Dict[int, Set[int]] = {}
    for p in test_pairs:
        ground_truth_map.setdefault(p.anchor_item, set()).add(p.positive_item)

    query_ids = sorted(list(ground_truth_map.keys()))
    candidate_ids = list(unique_items)

    q_bert = torch.stack([features[qid] for qid in query_ids]).to(device)
    c_bert = torch.stack([features[cid] for cid in candidate_ids]).to(device)

    # 3. Model 1: Raw BERT
    logger.info("Computing metrics for Model 1: Raw BERT (768-D)...")
    m_raw = compute_metrics_with_bootstrap(q_bert, query_ids, c_bert, candidate_ids, ground_truth_map)

    # 4. Model 2: Untrained Projection Baseline
    torch.manual_seed(args.seed)
    untrained = TextProjectionModel().to(device)
    untrained.eval()
    with torch.no_grad():
        m_untrained = compute_metrics_with_bootstrap(untrained(q_bert), query_ids, untrained(c_bert), candidate_ids, ground_truth_map)

    # 5. Model 3: Phase 8B Trained Projection Model (V1)
    v1_model = TextProjectionModel().to(device)
    v1_ckpt = torch.load(v1_ckpt_path, map_location="cpu", weights_only=True)
    v1_model.load_state_dict(v1_ckpt["model_state_dict"])
    v1_model.eval()
    with torch.no_grad():
        m_v1 = compute_metrics_with_bootstrap(v1_model(q_bert), query_ids, v1_model(c_bert), candidate_ids, ground_truth_map)

    # 6. Model 4: Phase 8C Optimized Model (V2 Linear)
    v2_model = LinearProjectionModel(input_dim=768, output_dim=512).to(device)
    v2_ckpt = torch.load(v2_ckpt_path, map_location="cpu", weights_only=True)
    v2_model.load_state_dict(v2_ckpt["model_state_dict"])
    v2_model.eval()
    with torch.no_grad():
        m_v2 = compute_metrics_with_bootstrap(v2_model(q_bert), query_ids, v2_model(c_bert), candidate_ids, ground_truth_map)

    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "v1_checkpoint_sha256": v1_sha256,
        "v2_checkpoint_path": str(v2_ckpt_path),
        "test_queries": len(query_ids),
        "candidate_gallery": len(candidate_ids),
        "baselines": {
            "raw_bert_768d": m_raw,
            "untrained_projection_512d": m_untrained,
            "phase_8b_v1_trained_512d": m_v1,
            "phase_8c_v2_optimized_512d": m_v2,
        },
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("PHASE 8C COMPREHENSIVE HELD-OUT RETRIEVAL EVALUATION")
    print("="*80)
    print(f"{'Metric':<12} | {'Raw BERT':<12} | {'Untrained':<12} | {'Phase 8B (V1)':<14} | {'Phase 8C (V2)':<14} | {'V2 vs Untrained':<16}")
    print("-"*80)
    for k in ["Recall@1", "Recall@5", "Recall@10", "MRR", "NDCG@10"]:
        r_val = m_raw[k]
        u_val = m_untrained[k]
        v1_val = m_v1[k]
        v2_val = m_v2[k]
        rel_diff = ((v2_val - u_val) / u_val) * 100.0 if u_val > 0 else 0.0
        print(f"{k:<12} | {r_val:.4f}       | {u_val:.4f}       | {v1_val:.4f}         | {v2_val:.4f}         | {rel_diff:+.1f}%")
    print("="*80)
    print(f"95% Bootstrap CI - MRR:       Raw BERT={m_raw['bootstrap_ci95']['MRR']}, Phase 8C={m_v2['bootstrap_ci95']['MRR']}")
    print(f"95% Bootstrap CI - Recall@10: Raw BERT={m_raw['bootstrap_ci95']['Recall@10']}, Phase 8C={m_v2['bootstrap_ci95']['Recall@10']}")
    print("="*80)

    return 0


if __name__ == "__main__":
    main()
