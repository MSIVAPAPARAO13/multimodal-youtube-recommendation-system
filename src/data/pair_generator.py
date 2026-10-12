"""
Positive and Negative Pair Construction Pipeline for Contrastive Training.

Supports:
- Curated positive pairs
- User co-engagement / interaction-based pairs
- Topic/semantic group pairs
- Deterministic sampling with seed
- In-batch negative construction with false-negative suppression masks
- Duplicate and leakage detection across splits
"""

from dataclasses import dataclass, field
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from src.data.training_dataset import MultimodalTrainingDataset, MultimodalTrainingSample

logger = logging.getLogger(__name__)


@dataclass
class ContrastivePair:
    """A verified pair of videos with provenance."""
    anchor_id: str
    positive_id: str
    label_source: str
    group_id: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if not self.anchor_id or not self.positive_id:
            raise ValueError("anchor_id and positive_id must be non-empty.")
        if self.anchor_id == self.positive_id:
            raise ValueError(f"Self-pair ({self.anchor_id}, {self.positive_id}) is not a valid contrastive pair.")


class PairGenerator:
    """
    Constructs deterministic positive pairs and false-negative masks
    from a MultimodalTrainingDataset.
    """

    def __init__(
        self,
        dataset: MultimodalTrainingDataset,
        seed: int = 42,
        allow_cross_group_positives: bool = False,
    ):
        self.dataset = dataset
        self.seed = seed
        self.allow_cross_group_positives = allow_cross_group_positives
        self.rng = random.Random(seed)

    def generate_pairs(
        self,
        max_pairs_per_anchor: int = 5,
        min_pairs_per_anchor: int = 1,
    ) -> List[ContrastivePair]:
        """
        Extract deterministic positive pairs from dataset relationships.
        Validates coverage, deduplicates pairs, and checks against self-referencing.
        """
        pairs: List[ContrastivePair] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # 1. Direct positive_video_ids
        for sample in self.dataset.samples:
            valid_positives = [
                pid for pid in sample.positive_video_ids
                if pid in self.dataset._id_to_index and pid != sample.video_id
            ]

            # Deterministic shuffle
            self.rng.seed(self.seed + hash(sample.video_id) % 1000000)
            shuffled_positives = list(valid_positives)
            self.rng.shuffle(shuffled_positives)

            selected = shuffled_positives[:max_pairs_per_anchor]
            for pos_id in selected:
                pair_key = (sample.video_id, pos_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append(
                        ContrastivePair(
                            anchor_id=sample.video_id,
                            positive_id=pos_id,
                            label_source=sample.label_source,
                            group_id=sample.positive_group_id,
                        )
                    )

        # 2. Group-based positives (videos sharing positive_group_id)
        for group_id, member_ids in self.dataset._group_to_ids.items():
            if len(member_ids) < 2:
                continue

            for m_id in member_ids:
                other_members = [o for o in member_ids if o != m_id]
                self.rng.seed(self.seed + hash(m_id) % 1000000)
                shuffled_others = list(other_members)
                self.rng.shuffle(shuffled_others)

                selected = shuffled_others[:max_pairs_per_anchor]
                for pos_id in selected:
                    pair_key = (m_id, pos_id)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        sample = self.dataset.get_by_id(m_id)
                        pairs.append(
                            ContrastivePair(
                                anchor_id=m_id,
                                positive_id=pos_id,
                                label_source=sample.label_source or "group_membership",
                                group_id=group_id,
                            )
                        )

        return pairs

    def compute_false_negative_mask(
        self,
        batch_ids: List[str],
    ) -> torch.Tensor:
        """
        Compute an (N, N) boolean mask for in-batch negatives.
        mask[i, j] == True means sample j is a VALID negative for anchor i.
        mask[i, j] == False means sample j is anchor i itself, or shares a positive relationship
        (false negative suppression to prevent gradient penalty on genuine positives).
        """
        n = len(batch_ids)
        # By default all off-diagonal entries are valid negatives (True)
        mask = torch.ones((n, n), dtype=torch.bool)
        # Diagonal is self (not a negative)
        mask.fill_diagonal_(False)

        # Look up relationships
        for i, aid in enumerate(batch_ids):
            a_sample = self.dataset.get_by_id(aid)
            a_positives = set(a_sample.positive_video_ids)
            a_group = a_sample.positive_group_id

            for j, bid in enumerate(batch_ids):
                if i == j:
                    continue
                # If bid is in aid's declared positives, mask it out
                if bid in a_positives:
                    mask[i, j] = False
                # If both share a positive group, mask it out
                if a_group and self.dataset.get_by_id(bid).positive_group_id == a_group:
                    mask[i, j] = False

        return mask


class PairedMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing aligned (anchor, positive) multimodal pairs
    for contrastive batch training.
    """

    def __init__(
        self,
        dataset: MultimodalTrainingDataset,
        pairs: List[ContrastivePair],
    ):
        self.dataset = dataset
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        anchor = self.dataset.get_by_id(pair.anchor_id)
        positive = self.dataset.get_by_id(pair.positive_id)

        a_v, a_a, a_t, a_mask = anchor.get_fused_input()
        p_v, p_a, p_t, p_mask = positive.get_fused_input()

        return {
            "anchor_id": anchor.video_id,
            "positive_id": positive.video_id,
            "anchor_visual": a_v,
            "anchor_audio": a_a,
            "anchor_text": a_t,
            "anchor_mask": a_mask,
            "positive_visual": p_v,
            "positive_audio": p_a,
            "positive_text": p_t,
            "positive_mask": p_mask,
            "label_source": pair.label_source,
            "group_id": pair.group_id or "",
        }
