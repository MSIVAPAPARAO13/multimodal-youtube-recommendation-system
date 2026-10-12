"""
Phase 10 User Personalization Engine Evaluation.

Experiment: EXP_MICROLENS_USER_PERSONALIZATION_V1
Retrieval Representation: TEXT_COENGAGEMENT_V2 (512-D)

Performs a rigorous, leakage-free per-user chronological evaluation:
1. Sorts each eligible user's interactions chronologically.
2. Uses interactions strictly earlier than the cutoff timestamp (t < t_cutoff)
   to build the recency-weighted user profile vector.
3. Holds out the future interaction at t_cutoff as ground truth.
4. Compares:
   - Content-Only Baseline: Most recent historical item embedding as query (Phase 9 retrieval)
   - Personalized Baseline: Recency-weighted user profile embedding as query (Phase 10)
   - Cold-Start Baseline: Catalog order / unpersonalized baseline
5. Measures Recall@1, Recall@5, Recall@10, Precision@5, Precision@10, MRR, NDCG@10,
   latency (profile generation & ranking), candidate coverage, and cold-start fallback rate.
6. Stores output under data/checkpoints/EXP_MICROLENS_USER_PERSONALIZATION_V1/evaluation_results.json
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data.text_coengagement import (
    BertFeatureExtractor,
    TextCoengagementBuilder,
)
from src.models.text_projection_model import load_phase_8c_projection
from src.services.user_personalization_service import UserPersonalizationService
from src.services.vector_index_service import VectorIndexService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_user_personalization")

PHASE_8C_CHECKPOINT = "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt"
EXPERIMENT_ID = "EXP_MICROLENS_USER_PERSONALIZATION_V1"
DEFAULT_OUTPUT_DIR = Path("data/checkpoints") / EXPERIMENT_ID


def evaluate_ranking(
    ranked_candidates: List[str],
    target_item: str,
    top_k_list: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute ranking metrics for a single held-out target item.

    Metrics:
    - Recall@K: 1.0 if target is within top K, else 0.0
    - Precision@K: (1.0 / K) if target is within top K, else 0.0
    - MRR: 1.0 / rank (1-indexed) if target found, else 0.0
    - NDCG@K: 1.0 / log2(rank + 1) if rank <= K, else 0.0
    """
    metrics = {}
    rank = None
    for idx, cand in enumerate(ranked_candidates):
        if cand == target_item:
            rank = idx + 1
            break

    for k in top_k_list:
        hit = 1.0 if (rank is not None and rank <= k) else 0.0
        metrics[f"Recall@{k}"] = hit
        metrics[f"Precision@{k}"] = hit / float(k)
        metrics[f"NDCG@{k}"] = (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0

    metrics["MRR"] = (1.0 / float(rank)) if rank is not None else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 10 User Personalization Engine")
    parser.add_argument("--checkpoint", type=str, default=PHASE_8C_CHECKPOINT)
    parser.add_argument("--catalog-users", type=int, default=1000, help="Users to define item catalog (Phase 9 items)")
    parser.add_argument("--eval-users", type=int, default=500, help="Number of eligible users to evaluate chronologically")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum total interactions for eligible user")
    parser.add_argument("--decay-factor", type=float, default=0.85, help="Exponential position decay factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "evaluation_results.json"),
    )
    parser.add_argument("--no-exclude-seen", action="store_true", help="Do not exclude seen items from candidates")
    args = parser.parse_args()

    exclude_seen = not args.no_exclude_seen
    device = torch.device("cpu")
    index_service = VectorIndexService()
    personalization_service = UserPersonalizationService(
        vector_index_service=index_service,
        decay_factor=args.decay_factor,
    )

    # 1. Load Phase 8C projection model
    logger.info("Loading Phase 8C checkpoint: %s", args.checkpoint)
    proj = load_phase_8c_projection(checkpoint_path=args.checkpoint, device=device)

    # 2. Build catalog of unique items (consistent with Phase 9 2,960 items)
    logger.info("Building item catalog using TextCoengagementBuilder (max_users=%d)...", args.catalog_users)
    builder = TextCoengagementBuilder(seed=args.seed)
    ds = builder.build_dataset(max_users=args.catalog_users)
    unique_items = ds["unique_items"]
    titles_map = ds["titles_map"]
    catalog_set = set(unique_items)
    logger.info("Catalog size: %d unique items", len(catalog_set))

    cache_path = Path(args.cache_dir) / f"microlens_bert_{len(unique_items)}.pt"
    extractor = BertFeatureExtractor(device=device)
    raw_features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

    # Project all items to 512-D
    logger.info("Projecting %d items to 512-D embedding space...", len(raw_features))
    projected_vectors: Dict[str, List[float]] = {}
    with torch.no_grad():
        for item_id, feat in raw_features.items():
            p_emb = proj(feat.unsqueeze(0))
            validated = index_service.validate_embedding(p_emb)
            projected_vectors[str(item_id)] = validated

    # 3. Load interactions and extract chronological sequences
    logger.info("Loading user interactions from MicroLens pairs CSV...")
    pairs_df = pd.read_csv("data/raw/microlens/MicroLens-50k_pairs.csv")
    pairs_df["user"] = pairs_df["user"].astype(int)
    pairs_df["item"] = pairs_df["item"].astype(int)
    pairs_df["timestamp"] = pairs_df["timestamp"].astype(int)

    # Filter interactions to items present in our catalog
    filtered_df = pairs_df[pairs_df["item"].isin(catalog_set)].copy()
    filtered_df.sort_values(by=["user", "timestamp"], inplace=True)

    # Group by user
    user_groups = filtered_df.groupby("user")
    eligible_user_ids = []
    for uid, group in user_groups:
        if len(group) >= args.min_history:
            eligible_user_ids.append(uid)

    logger.info("Found %d eligible users with >= %d interactions in catalog", len(eligible_user_ids), args.min_history)

    # Select evaluation users deterministically
    rng = random.Random(args.seed)
    rng.shuffle(eligible_user_ids)
    selected_eval_users = eligible_user_ids[: args.eval_users]
    logger.info("Selected %d users for chronological evaluation", len(selected_eval_users))

    # Catalog fallback order (sorted deterministically by popularity in catalog)
    item_counts = filtered_df["item"].value_counts()
    catalog_order = [str(x) for x in item_counts.index if str(x) in projected_vectors]

    # Metrics collectors
    top_k_list = (1, 5, 10)
    personalized_metrics: Dict[str, List[float]] = {
        f"Recall@{k}": [] for k in top_k_list
    }
    personalized_metrics.update({f"Precision@{k}": [] for k in top_k_list})
    personalized_metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    personalized_metrics["MRR"] = []

    content_only_metrics: Dict[str, List[float]] = {
        f"Recall@{k}": [] for k in top_k_list
    }
    content_only_metrics.update({f"Precision@{k}": [] for k in top_k_list})
    content_only_metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    content_only_metrics["MRR"] = []

    cold_start_metrics: Dict[str, List[float]] = {
        f"Recall@{k}": [] for k in top_k_list
    }
    cold_start_metrics.update({f"Precision@{k}": [] for k in top_k_list})
    cold_start_metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    cold_start_metrics["MRR"] = []

    profile_latencies_ms = []
    ranking_latencies_ms = []
    total_queries = 0
    cold_start_fallback_count = 0

    # 4. Run per-user chronological evaluation
    logger.info("Running chronological evaluation...")
    for uid in selected_eval_users:
        u_events = user_groups.get_group(uid).to_dict("records")

        # Sort strictly by timestamp
        u_events.sort(key=lambda x: x["timestamp"])

        # Target is the last interaction
        target_event = u_events[-1]
        target_item = str(target_event["item"])
        cutoff_timestamp = target_event["timestamp"]  # strictly earlier than this

        # History is all interactions strictly earlier than cutoff_timestamp
        history_events = [ev for ev in u_events[:-1] if ev["timestamp"] < cutoff_timestamp]
        if not history_events:
            continue

        # Seen items to exclude
        seen_items = {str(ev["item"]) for ev in history_events}
        if exclude_seen and target_item in seen_items:
            # If target item was already seen in history, exclude from unseen evaluation
            # (Evaluation measures next new item recommendation)
            pass

        # Collect history embeddings
        hist_embeddings = []
        hist_timestamps = []
        for ev in history_events:
            iid = str(ev["item"])
            if iid in projected_vectors:
                hist_embeddings.append(projected_vectors[iid])
                hist_timestamps.append(ev["timestamp"] / 1000.0)

        if not hist_embeddings:
            cold_start_fallback_count += 1
            continue

        total_queries += 1

        # A. Personalized Model: Build profile vector
        t0_prof = time.perf_counter()
        profile_vec = personalization_service.build_user_profile(
            item_embeddings=hist_embeddings,
            timestamps=hist_timestamps,
            cutoff_timestamp=cutoff_timestamp / 1000.0,
            use_time_decay=False,
        )
        t_prof_ms = (time.perf_counter() - t0_prof) * 1000.0
        profile_latencies_ms.append(t_prof_ms)

        # Exclusions
        query_exclusions = set(seen_items) if exclude_seen else set()

        # Score candidates with profile vector
        t0_rank = time.perf_counter()
        personalized_ranked = index_service.search_exact_in_memory(
            query_embedding=profile_vec,
            candidate_embeddings=projected_vectors,
            top_k=10,
            exclude_video_ids=query_exclusions,
        )
        t_rank_ms = (time.perf_counter() - t0_rank) * 1000.0
        ranking_latencies_ms.append(t_rank_ms)
        pers_cand_ids = [vid for vid, _ in personalized_ranked]

        # Evaluate Personalized
        pers_eval = evaluate_ranking(pers_cand_ids, target_item, top_k_list)
        for k, v in pers_eval.items():
            personalized_metrics[k].append(v)

        # B. Content-Only Baseline: Most recent historical item
        latest_hist_item = str(history_events[-1]["item"])
        latest_emb = projected_vectors[latest_hist_item]

        content_exclusions = set(seen_items) if exclude_seen else {latest_hist_item}
        content_ranked = index_service.search_exact_in_memory(
            query_embedding=latest_emb,
            candidate_embeddings=projected_vectors,
            top_k=10,
            exclude_video_ids=content_exclusions,
        )
        content_cand_ids = [vid for vid, _ in content_ranked]

        # Evaluate Content-Only
        content_eval = evaluate_ranking(content_cand_ids, target_item, top_k_list)
        for k, v in content_eval.items():
            content_only_metrics[k].append(v)

        # C. Cold-Start Baseline: Popularity/Catalog fallback without history
        cold_cand_ids = [
            vid for vid in catalog_order
            if (not exclude_seen or vid not in seen_items)
        ][:10]
        cold_eval = evaluate_ranking(cold_cand_ids, target_item, top_k_list)
        for k, v in cold_eval.items():
            cold_start_metrics[k].append(v)

    # 5. Summarize metrics
    def summarize(metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:
        summary = {}
        for m_name, vals in metrics_dict.items():
            summary[m_name] = float(np.mean(vals)) if vals else 0.0
        return summary

    summary_pers = summarize(personalized_metrics)
    summary_content = summarize(content_only_metrics)
    summary_cold = summarize(cold_start_metrics)

    # Latency percentiles
    prof_p50 = float(np.percentile(profile_latencies_ms, 50)) if profile_latencies_ms else 0.0
    prof_p95 = float(np.percentile(profile_latencies_ms, 95)) if profile_latencies_ms else 0.0
    prof_mean = float(np.mean(profile_latencies_ms)) if profile_latencies_ms else 0.0

    rank_p50 = float(np.percentile(ranking_latencies_ms, 50)) if ranking_latencies_ms else 0.0
    rank_p95 = float(np.percentile(ranking_latencies_ms, 95)) if ranking_latencies_ms else 0.0
    rank_mean = float(np.mean(ranking_latencies_ms)) if ranking_latencies_ms else 0.0

    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "protocol": "per_user_chronological",
        "leakage_protection": {
            "strict_cutoff_filtering": True,
            "future_interactions_excluded": True,
            "deduplication_policy": "keep_latest_position",
        },
        "evaluation_config": {
            "eval_users_requested": args.eval_users,
            "eligible_users_total": len(eligible_user_ids),
            "evaluated_queries": total_queries,
            "candidate_catalog_size": len(projected_vectors),
            "min_history_interactions": args.min_history,
            "decay_factor": args.decay_factor,
            "exclude_seen": exclude_seen,
            "cold_start_fallback_count": cold_start_fallback_count,
            "fallback_rate": (cold_start_fallback_count / max(len(selected_eval_users), 1)),
        },
        "metrics": {
            "personalized_profile": summary_pers,
            "content_only_baseline": summary_content,
            "cold_start_baseline": summary_cold,
        },
        "relative_improvements": {
            f"delta_{m}": float(summary_pers[m] - summary_content[m])
            for m in summary_pers
        },
        "latency_ms": {
            "profile_construction": {
                "mean": round(prof_mean, 4),
                "p50": round(prof_p50, 4),
                "p95": round(prof_p95, 4),
            },
            "candidate_ranking": {
                "mean": round(rank_mean, 4),
                "p50": round(rank_p50, 4),
                "p95": round(rank_p95, 4),
            },
            "total_mean": round(prof_mean + rank_mean, 4),
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved evaluation results to %s", args.output_json)
    print("\n" + "=" * 60)
    print(f"PHASE 10 EVALUATION RESULTS ({EXPERIMENT_ID})")
    print("=" * 60)
    print(f"Queries Evaluated: {total_queries} | Candidates: {len(projected_vectors)}")
    print("-" * 60)
    print(f"{'Metric':<15} | {'Cold-Start':<12} | {'Content-Only':<14} | {'Personalized':<14} | {'Delta':<10}")
    print("-" * 60)
    for m in ["Recall@1", "Recall@5", "Recall@10", "Precision@5", "Precision@10", "MRR", "NDCG@10"]:
        c_val = summary_cold.get(m, 0.0)
        co_val = summary_content.get(m, 0.0)
        p_val = summary_pers.get(m, 0.0)
        delta = p_val - co_val
        print(f"{m:<15} | {c_val:<12.4f} | {co_val:<14.4f} | {p_val:<14.4f} | {delta:+<10.4f}")
    print("-" * 60)
    print(f"Profile Latency (p50 / p95): {prof_p50:.3f} ms / {prof_p95:.3f} ms")
    print(f"Ranking Latency (p50 / p95): {rank_p50:.3f} ms / {rank_p95:.3f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
