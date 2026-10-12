"""
Loss Functions for Multimodal Contrastive Learning.

Implements:
1. InfoNCELoss (with temperature scaling, L2 normalization, false-negative suppression mask)
2. SymmetricContrastiveLoss (bidirectional anchor <-> positive)
3. SupervisedContrastiveLoss (multi-positive contrastive objective)
4. TripletLoss (margin-based comparison)

Ensures finite gradients, zero-vector rejection, and numerical stability.

Data Integrity Note:
Contrastive recommendation learning strictly requires genuine positive pairs derived from
user co-engagement sessions, watch sequences, or verified related videos. Artificial
Gaussian noise perturbation (e.g. positive = emb + noise) does not constitute genuine
recommendation relevance and is prohibited from being presented as valid model training.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    r"""
    InfoNCE (Noise Contrastive Estimation) Loss with false-negative suppression.

    Mathematical formulation:
    .. math::
        \mathcal{L}_i = -\log \frac{\exp(s(q_i, k_i^+) / \tau)}{\exp(s(q_i, k_i^+) / \tau) + \sum_{j \in \mathcal{N}_i} \exp(s(q_i, k_j^-) / \tau)}

    where:
    - :math:`s(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}` is cosine similarity
    - :math:`\tau` is the temperature parameter (:math:`\tau > 0`)
    - :math:`k_i^+` is the genuine positive pair for anchor :math:`q_i`
    - :math:`\mathcal{N}_i` is the set of valid in-batch negatives (controlled by negative_mask)

    Args:
        temperature: Scaling factor (must be > 0.0). Default: 0.07.
        normalize: Whether to explicitly L2-normalize input embeddings. Default: True.
    """

    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)
        self.normalize = normalize

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss over batch.

        Args:
            anchor_emb: (batch_size, dim) anchor embeddings
            positive_emb: (batch_size, dim) positive paired embeddings
            negative_mask: Optional (batch_size, batch_size) boolean tensor.
                           False indicates index (i, j) should NOT be used as negative.
            valid_mask: Optional (batch_size,) boolean tensor indicating which samples
                        have valid representations (e.g. non-zero embeddings).

        Returns:
            Scalar loss tensor.
        """
        if anchor_emb.size(0) == 0:
            return torch.tensor(0.0, device=anchor_emb.device, requires_grad=True)

        batch_size = anchor_emb.size(0)
        device = anchor_emb.device

        if self.normalize:
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)

        # Detect and mask zero-norm / missing vectors
        if valid_mask is None:
            a_valid = (torch.norm(anchor_emb, p=2, dim=-1) > 1e-6)
            p_valid = (torch.norm(positive_emb, p=2, dim=-1) > 1e-6)
            valid_mask = a_valid & p_valid

        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarities scaled by temperature: (B, B)
        similarity_matrix = (anchor_emb @ positive_emb.T) / self.temperature

        # Clamp similarities for numerical stability against overflow in exp
        similarity_matrix = torch.clamp(similarity_matrix, min=-50.0, max=50.0)

        # If negative_mask is provided, mask invalid negatives with large negative value
        if negative_mask is not None:
            # We want off-diagonal elements that are False in negative_mask to not contribute to denominator
            # Diagonal is the positive: similarity_matrix[i, i] is positive
            neg_suppression = ~negative_mask
            # Don't mask out diagonal (positive target)
            neg_suppression.fill_diagonal_(False)
            similarity_matrix = similarity_matrix.masked_fill(neg_suppression, -1e4)

        # Labels: diagonal elements are the positive matches (0 -> 0, 1 -> 1, ...)
        labels = torch.arange(batch_size, device=device)

        # Only compute loss on valid samples
        if not valid_mask.all():
            similarity_matrix = similarity_matrix[valid_mask]
            labels = labels[valid_mask]

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss (bidirectional InfoNCE).
    Computes loss from Anchor -> Positive AND Positive -> Anchor.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)
        self.symmetric = symmetric
        self.infonce = InfoNCELoss(temperature=self.temperature, normalize=normalize)

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.infonce(emb1, emb2, negative_mask=negative_mask, valid_mask=valid_mask)
        if self.symmetric:
            neg_mask_t = negative_mask.T if negative_mask is not None else None
            loss = (loss + self.infonce(emb2, emb1, negative_mask=neg_mask_t, valid_mask=valid_mask)) / 2.0
        return loss


class SupervisedContrastiveLoss(nn.Module):
    r"""
    Supervised Contrastive Learning (SupCon) Loss.

    Handles multiple positive samples per anchor (e.g., from the same topic or co-engagement group).
    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)
        self.normalize = normalize

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch_size, dim) or (batch_size, num_views, dim)
            labels: (batch_size,) class / group identifiers
            mask: Optional (batch_size, batch_size) binary mask where mask[i, j] == 1 if positive.
        """
        device = features.device
        if features.dim() == 3:
            # (B, V, D) -> stack into (B * V, D)
            b, v, d = features.shape
            features = features.view(b * v, d)
            labels = labels.contiguous().view(-1, 1).repeat(1, v).view(-1)

        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Compute dot products / similarities
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        sim_matrix = torch.clamp(sim_matrix, min=-50.0, max=50.0)

        # Subtract max for numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Compute log-probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positives
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero for anchors with no other positives in batch
        valid_anchors = mask_pos_pairs > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / mask_pos_pairs[valid_anchors]
        loss = -mean_log_prob_pos.mean()
        return loss


class TripletLoss(nn.Module):
    """
    Standard margin Triplet Loss: max(0, sim(a, n) - sim(a, p) + margin).
    """

    def __init__(self, margin: float = 0.3, normalize: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize:
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            negative_emb = F.normalize(negative_emb, p=2, dim=1)

        pos_sim = (anchor_emb * positive_emb).sum(dim=1)
        neg_sim = (anchor_emb * negative_emb).sum(dim=1)

        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()
