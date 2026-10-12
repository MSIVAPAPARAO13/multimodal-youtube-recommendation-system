"""
Optimization and Diagnostics Experiment Runner.

Experiment ID: EXP_MICROLENS_TEXT_COENGAGEMENT_V2
Evaluates candidate projection architectures, learning rates, temperatures, and selection criteria
STRICTLY on the Validation set. Test set is NEVER touched during optimization.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.text_coengagement import (
    TextCoengagementBuilder,
    BertFeatureExtractor,
    PairedTextFeatureDataset,
)
from src.models.text_projection_model import TextProjectionModel
from src.training.losses import ContrastiveLoss
from scripts.evaluate_text_coengagement import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optimize_text_coengagement")


class LinearProjectionModel(nn.Module):
    """Direct single-layer linear projection without non-linear ReLU distortion."""
    def __init__(self, input_dim: int = 768, output_dim: int = 512, normalize: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.linear(x)
        if self.normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        return proj


class ResidualProjectionModel(nn.Module):
    """Projection with residual skip connection to preserve pretrained semantic geometry."""
    def __init__(self, input_dim: int = 768, output_dim: int = 512, dropout: float = 0.1, normalize: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        # Direct projection path
        self.direct = nn.Linear(input_dim, output_dim)
        # Non-linear refinement path
        self.refinement = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.direct(x) + 0.5 * self.refinement(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


def train_and_eval_validation(
    model_cls: type,
    model_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_query_tensor: torch.Tensor,
    val_query_ids: List[int],
    val_cand_tensor: torch.Tensor,
    val_cand_ids: List[int],
    val_ground_truth: dict,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    epochs: int = 5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    model = model_cls(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(len(train_loader) * epochs, 1), eta_min=lr * 0.01)
    criterion = ContrastiveLoss(temperature=temperature, symmetric=True)

    best_val_loss = float("inf")
    best_val_mrr = 0.0
    best_epoch = 0
    best_state = None

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            a_feat = batch["anchor_feature"].to(device)
            p_feat = batch["positive_feature"].to(device)

            z_a = model(a_feat)
            z_p = model(p_feat)

            loss = criterion(z_a, z_p)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)

        # Validation loss
        model.eval()
        v_total_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                a_feat = batch["anchor_feature"].to(device)
                p_feat = batch["positive_feature"].to(device)
                z_a = model(a_feat)
                z_p = model(p_feat)
                v_loss = criterion(z_a, z_p)
                v_total_loss += v_loss.item()
                v_batches += 1

            val_loss = v_total_loss / max(v_batches, 1)

            # Validation retrieval ranking metrics
            q_emb = model(val_query_tensor)
            c_emb = model(val_cand_tensor)
            val_metrics = compute_metrics(q_emb, val_query_ids, c_emb, val_cand_ids, val_ground_truth)

        val_mrr = val_metrics["MRR"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mrr": val_mrr,
            "val_recall1": val_metrics["Recall@1"],
            "val_recall5": val_metrics["Recall@5"],
            "val_ndcg10": val_metrics["NDCG@10"],
        })

        # Track best by validation retrieval ranking (MRR & NDCG)
        if val_mrr > best_val_mrr or (val_mrr == best_val_mrr and val_loss < best_val_loss):
            best_val_mrr = val_mrr
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mrr": best_val_mrr,
        "best_state": best_state,
        "history": history,
        "final_val_metrics": history[best_epoch - 1] if best_epoch > 0 else history[-1],
    }


def main():
    device = torch.device("cpu")
    seed = 42
    torch.manual_seed(seed)

    # Load dataset
    builder = TextCoengagementBuilder(seed=seed)
    ds = builder.build_dataset(max_users=1000)
    unique_items = ds["unique_items"]
    titles_map = ds["titles_map"]

    cache_path = Path(f"data/cache/microlens_bert_{len(unique_items)}.pt")
    extractor = BertFeatureExtractor(device=device)
    features = extractor.extract_features(unique_items, titles_map, cache_file=cache_path)

    train_ds = PairedTextFeatureDataset(ds["train_pairs"], features)
    val_ds = PairedTextFeatureDataset(ds["val_pairs"], features)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    val_ground_truth = {}
    for p in ds["val_pairs"]:
        val_ground_truth.setdefault(p.anchor_item, set()).add(p.positive_item)

    val_query_ids = sorted(list(val_ground_truth.keys()))
    val_cand_ids = list(unique_items)

    val_query_tensor = torch.stack([features[qid] for qid in val_query_ids]).to(device)
    val_cand_tensor = torch.stack([features[cid] for cid in val_cand_ids]).to(device)

    # 1. Baseline measurements on Validation Set
    raw_m = compute_metrics(val_query_tensor, val_query_ids, val_cand_tensor, val_cand_ids, val_ground_truth)
    logger.info("VALIDATION BASELINE - Raw BERT: MRR=%.4f, R@1=%.4f, R@5=%.4f, NDCG@10=%.4f", raw_m["MRR"], raw_m["Recall@1"], raw_m["Recall@5"], raw_m["NDCG@10"])

    untrained_2l = TextProjectionModel().to(device)
    with torch.no_grad():
        un_m = compute_metrics(untrained_2l(val_query_tensor), val_query_ids, untrained_2l(val_cand_tensor), val_cand_ids, val_ground_truth)
    logger.info("VALIDATION BASELINE - Untrained 2-layer MLP: MRR=%.4f, R@1=%.4f, R@5=%.4f, NDCG@10=%.4f", un_m["MRR"], un_m["Recall@1"], un_m["Recall@5"], un_m["NDCG@10"])

    untrained_lin = LinearProjectionModel().to(device)
    with torch.no_grad():
        un_lin_m = compute_metrics(untrained_lin(val_query_tensor), val_query_ids, untrained_lin(val_cand_tensor), val_cand_ids, val_ground_truth)
    logger.info("VALIDATION BASELINE - Untrained Linear: MRR=%.4f, R@1=%.4f, R@5=%.4f, NDCG@10=%.4f", un_lin_m["MRR"], un_lin_m["Recall@1"], un_lin_m["Recall@5"], un_lin_m["NDCG@10"])

    # 2. Experiment Grid on Validation Set
    experiments = [
        # Exp 1: Baseline Phase 8B setup (2-layer MLP, lr=1e-4, temp=0.07)
        ("mlp_lr1e4_t007", TextProjectionModel, {}, 1e-4, 1e-5, 0.07),
        # Exp 2: 2-layer MLP with lower lr (3e-5)
        ("mlp_lr3e5_t007", TextProjectionModel, {}, 3e-5, 1e-5, 0.07),
        # Exp 3: 2-layer MLP with lr=5e-5, temp=0.05
        ("mlp_lr5e5_t005", TextProjectionModel, {}, 5e-5, 1e-5, 0.05),
        # Exp 4: Linear projection (simpler architecture), lr=1e-4, temp=0.07
        ("linear_lr1e4_t007", LinearProjectionModel, {}, 1e-4, 1e-5, 0.07),
        # Exp 5: Linear projection, lr=3e-4, temp=0.07
        ("linear_lr3e4_t007", LinearProjectionModel, {}, 3e-4, 1e-5, 0.07),
        # Exp 6: Linear projection, lr=1e-3, temp=0.07
        ("linear_lr1e3_t007", LinearProjectionModel, {}, 1e-3, 1e-5, 0.07),
        # Exp 7: Residual projection head, lr=1e-4, temp=0.07
        ("residual_lr1e4_t007", ResidualProjectionModel, {}, 1e-4, 1e-5, 0.07),
        # Exp 8: Residual projection head, lr=3e-5, temp=0.07
        ("residual_lr3e5_t007", ResidualProjectionModel, {}, 3e-5, 1e-5, 0.07),
    ]

    results = {}
    best_candidate_name = None
    best_candidate_score = 0.0
    best_candidate_result = None

    for name, m_cls, m_kw, lr, wd, temp in experiments:
        torch.manual_seed(seed)
        res = train_and_eval_validation(
            model_cls=m_cls,
            model_kwargs=m_kw,
            train_loader=train_loader,
            val_loader=val_loader,
            val_query_tensor=val_query_tensor,
            val_query_ids=val_query_ids,
            val_cand_tensor=val_cand_tensor,
            val_cand_ids=val_cand_ids,
            val_ground_truth=val_ground_truth,
            lr=lr,
            weight_decay=wd,
            temperature=temp,
            epochs=5,
            device=device,
        )
        fm = res["final_val_metrics"]
        logger.info(
            "[%s] Best Epoch: %d - Val Loss: %.4f - MRR: %.4f - R@1: %.4f - R@5: %.4f - NDCG@10: %.4f",
            name, res["best_epoch"], res["best_val_loss"], fm["val_mrr"], fm["val_recall1"], fm["val_recall5"], fm["val_ndcg10"]
        )
        results[name] = {
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "val_metrics": fm,
            "architecture": m_cls.__name__,
            "lr": lr,
            "temperature": temp,
        }

        # Track best by validation MRR + NDCG
        val_score = fm["val_mrr"] + fm["val_ndcg10"]
        if val_score > best_candidate_score:
            best_candidate_score = val_score
            best_candidate_name = name
            best_candidate_result = res

    logger.info("="*60)
    logger.info("OPTIMIZATION COMPLETED ON VALIDATION SET")
    logger.info("Selected Best Candidate: %s (Score: %.4f)", best_candidate_name, best_candidate_score)
    logger.info("="*60)

    # Save validation tuning report
    out_dir = Path("data/checkpoints/EXP_MICROLENS_TEXT_COENGAGEMENT_V2")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "validation_optimization_matrix.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_baselines": {
                "raw_bert": raw_m,
                "untrained_2layer_mlp": un_m,
                "untrained_linear": un_lin_m,
            },
            "grid_results": results,
            "selected_candidate": best_candidate_name,
        }, f, indent=2)

    return best_candidate_name, best_candidate_result


if __name__ == "__main__":
    main()
