"""
Tests for Evaluation Protocol (Recall@K, MRR, NDCG@K).
Verifies:
- Metric computation accuracy
- Handling of queries without relevant candidates
- Candidate set separation
"""

import pytest
import torch

from scripts.evaluate_multimodal import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    evaluate_retrieval,
)


def test_retrieval_metrics_calculation():
    ground_truth = {"item_1", "item_2"}
    ranked_list = ["item_1", "item_3", "item_2", "item_4"]

    # Recall@1: 1 hit out of 2 ground truth -> 0.5
    assert compute_recall_at_k(ranked_list, ground_truth, k=1) == pytest.approx(0.5)
    # Recall@3: 2 hits out of 2 ground truth -> 1.0
    assert compute_recall_at_k(ranked_list, ground_truth, k=3) == pytest.approx(1.0)

    # Precision@1: 1 hit out of 1 -> 1.0
    assert compute_precision_at_k(ranked_list, ground_truth, k=1) == pytest.approx(1.0)
    # Precision@2: 1 hit out of 2 -> 0.5
    assert compute_precision_at_k(ranked_list, ground_truth, k=2) == pytest.approx(0.5)

    # MRR: first hit is at rank 1 -> 1.0
    assert compute_mrr(ranked_list, ground_truth) == pytest.approx(1.0)

    # MRR when first hit is at rank 2
    assert compute_mrr(["item_3", "item_1"], ground_truth) == pytest.approx(0.5)

    # NDCG@K
    assert compute_ndcg_at_k(ranked_list, ground_truth, k=3) > 0.0


def test_evaluate_retrieval_batch():
    query_embeddings = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    query_ids = ["q1", "q2"]

    candidate_embeddings = torch.tensor([
        [1.0, 0.0],  # matches q1
        [0.0, 1.0],  # matches q2
        [0.5, 0.5],
    ])
    candidate_ids = ["c1", "c2", "c3"]

    ground_truth_map = {
        "q1": {"c1"},
        "q2": {"c2"},
    }

    res = evaluate_retrieval(
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        candidate_embeddings=candidate_embeddings,
        candidate_ids=candidate_ids,
        ground_truth_map=ground_truth_map,
        top_k_list=(1, 2),
    )

    assert res["valid_queries"] == 2
    assert res["Recall@1"] == pytest.approx(1.0)
    assert res["MRR"] == pytest.approx(1.0)


def test_queries_without_relevant_candidates():
    query_embeddings = torch.randn(2, 8)
    query_ids = ["q1", "q2"]
    candidate_embeddings = torch.randn(3, 8)
    candidate_ids = ["c1", "c2", "c3"]

    # Empty ground truth for all queries
    ground_truth_map = {"q1": set(), "q2": set()}

    res = evaluate_retrieval(
        query_embeddings,
        query_ids,
        candidate_embeddings,
        candidate_ids,
        ground_truth_map,
    )
    assert res["valid_queries"] == 0
