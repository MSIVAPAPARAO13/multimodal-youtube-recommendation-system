"""
Phase 11A Approximate Nearest Neighbor (ANN) Retrieval Benchmark.

Experiment: EXP_MICROLENS_ANN_RETRIEVAL_V1
Retrieval Representation: TEXT_COENGAGEMENT_V2 (512-D, Phase 8C checkpoint)
Database Backend: PostgreSQL 16 with pgvector 0.8.7 (HNSW cosine distance index)

Benchmarks:
1. Exact Cosine Similarity vs pgvector HNSW ANN retrieval across identical queries and catalog.
2. Measures ANN agreement (Recall@1, Recall@5, Recall@10, Recall@50 relative to exact search).
3. Measures latency percentiles (p50, p95, p99, mean) and throughput (QPS).
4. Tests both item-to-item query vectors and Phase 10 user profile vectors.
5. Saves results to data/checkpoints/EXP_MICROLENS_ANN_RETRIEVAL_V1/benchmark_results.json.
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
import torch

from src.db.config import get_settings
from src.db.database import Database
from src.db.models import Video
from src.db.repository import VideoRepository
from src.models.text_projection_model import load_phase_8c_projection
from src.services.vector_index_service import VectorIndexService
from src.services.user_personalization_service import UserPersonalizationService
from src.data.text_coengagement import (
    BertFeatureExtractor,
    TextCoengagementBuilder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark_ann_retrieval")

PHASE_8C_CHECKPOINT = "data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2/best.pt"
EXPERIMENT_ID = "EXP_MICROLENS_ANN_RETRIEVAL_V1"
DEFAULT_OUTPUT_DIR = Path("data/checkpoints") / EXPERIMENT_ID


def compute_ann_recall(exact_ids: List[str], ann_ids: List[str], k: int) -> float:
    """
    Compute ANN Recall@K relative to exact search ground truth.

    Formula:
        Recall@K = |Exact_top_K ∩ ANN_top_K| / K
    """
    if k <= 0:
        return 0.0
    exact_k = set(exact_ids[:k])
    ann_k = set(ann_ids[:k])
    if not exact_k:
        return 1.0
    overlap = len(exact_k.intersection(ann_k))
    return overlap / float(len(exact_k))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 11A ANN retrieval against exact baseline")
    parser.add_argument("--checkpoint", type=str, default=PHASE_8C_CHECKPOINT)
    parser.add_argument("--catalog-users", type=int, default=1000, help="Users to define catalog")
    parser.add_argument("--num-queries", type=int, default=300, help="Number of benchmark queries to evaluate")
    parser.add_argument("--ef-search", type=int, default=40, help="HNSW ef_search parameter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=15, help="Number of warm-up queries")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "benchmark_results.json"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    index_service = VectorIndexService(ef_search=args.ef_search)

    # 1. Connect to PostgreSQL database
    logger.info("Connecting to database...")
    db = Database()
    session = db.SessionLocal()
    repo = VideoRepository(session)

    # Verify catalog count
    catalog_count = session.query(Video).filter(Video.embedding.isnot(None)).count()
    logger.info("Total embedded videos in database: %d", catalog_count)
    if catalog_count < 100:
        raise RuntimeError(
            f"Insufficient videos in database ({catalog_count}). Run scripts/build_vector_index.py --from-microlens-catalog first."
        )

    # Verify HNSW index existence in PostgreSQL
    from sqlalchemy import text
    index_res = session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'videos' AND indexname = 'ix_video_embedding';")
    ).fetchall()
    if not index_res:
        raise RuntimeError("HNSW index 'ix_video_embedding' not found on table 'videos'!")
    logger.info("Verified HNSW index: %s", index_res[0][1])

    # 2. Build test query vectors (combination of held-out item queries and user profiles)
    logger.info("Loading MicroLens dataset for test queries...")
    builder = TextCoengagementBuilder(seed=args.seed)
    ds = builder.build_dataset(max_users=args.catalog_users)
    unique_items = ds["unique_items"]
    titles_map = ds["titles_map"]

    cache_path = Path("data/cache") / f"microlens_bert_{len(unique_items)}.pt"
    extractor = BertFeatureExtractor(device=device)
    raw_features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

    proj = load_phase_8c_projection(checkpoint_path=args.checkpoint, device=device)

    # Project items into memory for fast exact reference
    projected_catalog: Dict[str, List[float]] = {}
    with torch.no_grad():
        for iid, feat in raw_features.items():
            p_emb = proj(feat.unsqueeze(0))
            validated = index_service.validate_embedding(p_emb)
            projected_catalog[str(iid)] = validated

    test_item_ids = list(projected_catalog.keys())
    random.shuffle(test_item_ids)

    # Prepare query vectors
    # 80% item-to-item vectors, 20% synthesized Phase 10 user profiles
    num_item_queries = int(args.num_queries * 0.8)
    num_user_queries = args.num_queries - num_item_queries

    queries: List[Tuple[str, List[float], Set[str]]] = []  # (query_type, query_vector, exclusions)

    # A. Item queries
    for vid in test_item_ids[:num_item_queries]:
        q_vec = projected_catalog[vid]
        queries.append(("item_seed", q_vec, {vid}))

    # B. Phase 10 User profile queries
    personalization_svc = UserPersonalizationService(vector_index_service=index_service)
    for _ in range(num_user_queries):
        hist_size = random.randint(2, 8)
        sample_vids = random.sample(test_item_ids, hist_size)
        sample_embs = [projected_catalog[v] for v in sample_vids]
        profile_vec = personalization_svc.build_user_profile(sample_embs)
        if profile_vec:
            queries.append(("user_profile", profile_vec, set(sample_vids)))

    logger.info("Generated %d benchmark queries (%d item queries, %d user profiles)", len(queries), num_item_queries, len(queries) - num_item_queries)

    # 3. Warm-up
    logger.info("Running %d warm-up queries on ANN and exact backends...", args.warmup)
    for _, q_vec, excl in queries[:args.warmup]:
        repo.find_similar_with_filters(
            embedding=q_vec, limit=10, exclude_video_ids=list(excl), backend="ann", ef_search=args.ef_search
        )
        repo.find_similar_with_filters(
            embedding=q_vec, limit=10, exclude_video_ids=list(excl), backend="exact"
        )

    # 4. Benchmark Execution
    k_eval_list = (1, 5, 10, 50)
    ann_recalls: Dict[str, List[float]] = {f"Recall@{k}": [] for k in k_eval_list}
    jaccard_top10: List[float] = []

    exact_latencies_ms: List[float] = []
    ann_latencies_ms: List[float] = []

    logger.info("Executing benchmark across %d queries...", len(queries))
    start_bench_time = time.perf_counter()

    for idx, (q_type, q_vec, exclusions) in enumerate(queries):
        excl_list = list(exclusions)

        # Exact Cosine Similarity (Ground Truth Reference)
        t0_exact = time.perf_counter()
        exact_results, exact_backend = repo.find_similar_with_filters(
            embedding=q_vec,
            limit=50,
            exclude_video_ids=excl_list,
            backend="exact",
            return_backend=True,
        )
        t_exact_ms = (time.perf_counter() - t0_exact) * 1000.0
        exact_latencies_ms.append(t_exact_ms)

        # ANN pgvector HNSW
        t0_ann = time.perf_counter()
        ann_results, ann_backend = repo.find_similar_with_filters(
            embedding=q_vec,
            limit=50,
            exclude_video_ids=excl_list,
            backend="ann",
            ef_search=args.ef_search,
            return_backend=True,
        )
        t_ann_ms = (time.perf_counter() - t0_ann) * 1000.0
        ann_latencies_ms.append(t_ann_ms)

        exact_ids = [v.video_id for v, _ in exact_results]
        ann_ids = [v.video_id for v, _ in ann_results]

        # Agreement metrics
        for k in k_eval_list:
            rec = compute_ann_recall(exact_ids, ann_ids, k)
            ann_recalls[f"Recall@{k}"].append(rec)

        # Top-10 Jaccard similarity
        s_exact_10 = set(exact_ids[:10])
        s_ann_10 = set(ann_ids[:10])
        union_10 = len(s_exact_10.union(s_ann_10))
        jacc = (len(s_exact_10.intersection(s_ann_10)) / float(union_10)) if union_10 > 0 else 1.0
        jaccard_top10.append(jacc)

    total_bench_time = time.perf_counter() - start_bench_time
    session.close()

    # 5. Compute summary statistics
    def calc_percentiles(arr: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
        }

    exact_lat_stats = calc_percentiles(exact_latencies_ms)
    ann_lat_stats = calc_percentiles(ann_latencies_ms)

    speedup_ratio = round(exact_lat_stats["mean"] / max(ann_lat_stats["mean"], 0.001), 2)
    ann_qps = round(len(queries) / (sum(ann_latencies_ms) / 1000.0), 1)
    exact_qps = round(len(queries) / (sum(exact_latencies_ms) / 1000.0), 1)

    recall_summary = {
        m_name: round(float(np.mean(vals)), 4)
        for m_name, vals in ann_recalls.items()
    }
    recall_summary["Jaccard@10"] = round(float(np.mean(jaccard_top10)), 4)

    # 6. Save JSON results
    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "database": {
            "engine": "PostgreSQL",
            "version": "16.15",
            "extension": "pgvector 0.8.7",
            "index_type": "HNSW (hierarchical navigable small world)",
            "distance_metric": "cosine (<=>)",
            "ef_search": args.ef_search,
            "catalog_size": catalog_count,
            "embedding_dimension": 512,
        },
        "benchmark_setup": {
            "num_queries": len(queries),
            "warmup_queries": args.warmup,
            "query_types": {
                "item_seed_queries": num_item_queries,
                "user_profile_queries": len(queries) - num_item_queries,
            },
        },
        "ann_agreement_vs_exact": recall_summary,
        "latency_ms": {
            "exact_search": exact_lat_stats,
            "ann_hnsw_search": ann_lat_stats,
            "speedup_factor": speedup_ratio,
        },
        "throughput_qps": {
            "exact_search": exact_qps,
            "ann_hnsw_search": ann_qps,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    logger.info("Saved benchmark results to %s", args.output_json)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"PHASE 11A ANN RETRIEVAL BENCHMARK ({EXPERIMENT_ID})")
    print("=" * 65)
    print(f"Catalog Size: {catalog_count} | Queries: {len(queries)} | ef_search: {args.ef_search}")
    print("-" * 65)
    print("ANN Retrieval Recall Relative to Exact Nearest Neighbors:")
    for k_metric in ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Jaccard@10"]:
        print(f"  {k_metric:<15}: {recall_summary[k_metric]:.4f} ({recall_summary[k_metric]*100.0:.2f}%)")
    print("-" * 65)
    print("Latency & Throughput Comparison:")
    print(f"  Exact Latency (mean / p50 / p95 / p99): {exact_lat_stats['mean']} ms / {exact_lat_stats['p50']} ms / {exact_lat_stats['p95']} ms / {exact_lat_stats['p99']} ms (QPS: {exact_qps})")
    print(f"  ANN Latency   (mean / p50 / p95 / p99): {ann_lat_stats['mean']} ms / {ann_lat_stats['p50']} ms / {ann_lat_stats['p95']} ms / {ann_lat_stats['p99']} ms (QPS: {ann_qps})")
    print(f"  Speedup Factor: {speedup_ratio}x faster with pgvector HNSW")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
