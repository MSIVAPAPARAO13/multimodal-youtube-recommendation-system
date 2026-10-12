"""
Multimodal Training Dataset Schema and Validation Pipeline.

Defines reproducible, typed schema for multimodal contrastive training,
strict tensor validation (512-dim representations, NaN/Inf rejection),
modality availability masks, and leak-free split management.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_SPLITS = {"train", "val", "test"}
DEFAULT_FEATURE_DIM = 512


@dataclass
class MultimodalTrainingSample:
    """
    Validated sample representation for multimodal contrastive training.
    """
    video_id: str
    visual_embedding: Optional[torch.Tensor] = None
    audio_embedding: Optional[torch.Tensor] = None
    text_embedding: Optional[torch.Tensor] = None
    title: Optional[str] = None
    positive_group_id: Optional[str] = None
    positive_video_ids: List[str] = field(default_factory=list)
    label_source: str = "curated"
    split: str = "train"
    modality_availability: Optional[List[bool]] = None
    feature_dim: int = DEFAULT_FEATURE_DIM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Execute strict schema, dimension, NaN/Inf, and split validation."""
        if not self.video_id or not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must be a non-empty string.")

        if self.split not in VALID_SPLITS:
            raise ValueError(f"split '{self.split}' must be one of {VALID_SPLITS}.")

        # Tensor dimension and numerical hygiene validation
        modalities = [
            ("visual", self.visual_embedding),
            ("audio", self.audio_embedding),
            ("text", self.text_embedding),
        ]
        computed_availability = []

        for name, emb in modalities:
            if emb is not None:
                if not isinstance(emb, torch.Tensor):
                    if isinstance(emb, (np.ndarray, list)):
                        emb = torch.tensor(emb, dtype=torch.float32)
                        if name == "visual":
                            self.visual_embedding = emb
                        elif name == "audio":
                            self.audio_embedding = emb
                        else:
                            self.text_embedding = emb
                    else:
                        raise TypeError(f"{name}_embedding must be a torch.Tensor, ndarray, or list.")

                # Ensure 1D tensor [feature_dim] or flattened [1, feature_dim]
                if emb.dim() == 2 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                    if name == "visual":
                        self.visual_embedding = emb
                    elif name == "audio":
                        self.audio_embedding = emb
                    else:
                        self.text_embedding = emb

                if emb.dim() != 1 or emb.shape[0] != self.feature_dim:
                    raise ValueError(
                        f"{name}_embedding must have shape ({self.feature_dim},), got {tuple(emb.shape)}"
                    )

                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    raise ValueError(f"{name}_embedding contains NaN or Inf values.")

                # Check if it is an all-zero dummy representation
                is_nonzero = (torch.norm(emb, p=2) > 1e-6).item()
                computed_availability.append(bool(is_nonzero))
            else:
                computed_availability.append(False)

        # Reconcile or auto-populate modality_availability
        if self.modality_availability is None:
            self.modality_availability = computed_availability
        else:
            if len(self.modality_availability) != 3:
                raise ValueError("modality_availability must have length 3 [visual, audio, text].")
            for i, (declared, actual) in enumerate(zip(self.modality_availability, computed_availability)):
                mod_name = ["visual", "audio", "text"][i]
                if declared and not actual:
                    raise ValueError(
                        f"Modality '{mod_name}' declared available but embedding is missing or zero vector."
                    )

        # Must have at least one modality available to be valid
        if not any(self.modality_availability):
            raise ValueError(f"Sample {self.video_id} has no valid modalities available.")

        # Ensure positive_video_ids is list of strings without self-reference
        if self.video_id in self.positive_video_ids:
            raise ValueError(f"Sample {self.video_id} cannot have itself in positive_video_ids.")

    def get_fused_input(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return tensors ready for GatedFusion or TwoTowerModel along with modality mask.
        Returns:
            (visual_tensor, audio_tensor, text_tensor, mask_tensor)
        """
        zero = torch.zeros(self.feature_dim, dtype=torch.float32)
        v = self.visual_embedding if self.visual_embedding is not None else zero
        a = self.audio_embedding if self.audio_embedding is not None else zero
        t = self.text_embedding if self.text_embedding is not None else zero
        mask = torch.tensor(self.modality_availability, dtype=torch.bool)
        return v, a, t, mask


class MultimodalTrainingDataset(Dataset):
    """
    In-memory validated multimodal training dataset supporting train/val/test splits,
    provenance tracking, and strict duplicate/leakage prevention.
    """

    def __init__(
        self,
        samples: Optional[List[MultimodalTrainingSample]] = None,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ):
        self.feature_dim = feature_dim
        self.samples: List[MultimodalTrainingSample] = []
        self._id_to_index: Dict[str, int] = {}
        self._group_to_ids: Dict[str, List[str]] = {}

        if samples:
            for s in samples:
                self.add_sample(s)

    def add_sample(self, sample: MultimodalTrainingSample) -> None:
        """Add a single validated sample, checking for duplicates."""
        if sample.video_id in self._id_to_index:
            raise ValueError(f"Duplicate video_id '{sample.video_id}' detected.")
        if sample.feature_dim != self.feature_dim:
            raise ValueError(
                f"Sample feature_dim ({sample.feature_dim}) does not match dataset feature_dim ({self.feature_dim})."
            )

        idx = len(self.samples)
        self.samples.append(sample)
        self._id_to_index[sample.video_id] = idx

        if sample.positive_group_id:
            self._group_to_ids.setdefault(sample.positive_group_id, []).append(sample.video_id)

    def get_by_id(self, video_id: str) -> MultimodalTrainingSample:
        """Retrieve sample by video_id."""
        if video_id not in self._id_to_index:
            raise KeyError(f"video_id '{video_id}' not found in dataset.")
        return self.samples[self._id_to_index[video_id]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        v, a, t, mask = sample.get_fused_input()
        return {
            "video_id": sample.video_id,
            "visual": v,
            "audio": a,
            "text": t,
            "mask": mask,
            "positive_group_id": sample.positive_group_id or "",
            "positive_video_ids": sample.positive_video_ids,
            "split": sample.split,
            "label_source": sample.label_source,
        }

    def filter_split(self, split: str) -> "MultimodalTrainingDataset":
        """Create a new dataset containing only samples from the specified split."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split '{split}' must be one of {VALID_SPLITS}.")
        filtered = [s for s in self.samples if s.split == split]
        return MultimodalTrainingDataset(samples=filtered, feature_dim=self.feature_dim)

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate dataset-wide consistency:
        - Check for split leakage (e.g. cross-split positive pairs).
        - Verify positive references exist.
        - Check distribution across splits.
        """
        split_counts = {s: 0 for s in VALID_SPLITS}
        all_ids = set(self._id_to_index.keys())

        train_ids: Set[str] = set()
        val_ids: Set[str] = set()
        test_ids: Set[str] = set()

        for s in self.samples:
            split_counts[s.split] += 1
            if s.split == "train":
                train_ids.add(s.video_id)
            elif s.split == "val":
                val_ids.add(s.video_id)
            elif s.split == "test":
                test_ids.add(s.video_id)

        # Check for ID collisions across splits
        leakage_train_val = train_ids.intersection(val_ids)
        leakage_train_test = train_ids.intersection(test_ids)
        leakage_val_test = val_ids.intersection(test_ids)

        if leakage_train_val or leakage_train_test or leakage_val_test:
            raise RuntimeError(
                f"Split leakage detected across dataset splits! "
                f"Train/Val: {len(leakage_train_val)}, Train/Test: {len(leakage_train_test)}"
            )

        # Check that positive pairs do not cross split boundaries
        cross_split_positives = 0
        missing_positive_refs = 0
        for s in self.samples:
            for pos_id in s.positive_video_ids:
                if pos_id not in all_ids:
                    missing_positive_refs += 1
                    continue
                pos_sample = self.get_by_id(pos_id)
                if pos_sample.split != s.split:
                    cross_split_positives += 1

        if cross_split_positives > 0:
            raise RuntimeError(
                f"Split leakage detected: {cross_split_positives} positive pairs span different splits!"
            )

        return {
            "total_samples": len(self.samples),
            "split_counts": split_counts,
            "cross_split_positives": cross_split_positives,
            "missing_positive_refs": missing_positive_refs,
            "total_groups": len(self._group_to_ids),
        }

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize dataset into JSON-serializable list of records."""
        records = []
        for s in self.samples:
            records.append({
                "video_id": s.video_id,
                "visual_embedding": s.visual_embedding.cpu().tolist() if s.visual_embedding is not None else None,
                "audio_embedding": s.audio_embedding.cpu().tolist() if s.audio_embedding is not None else None,
                "text_embedding": s.text_embedding.cpu().tolist() if s.text_embedding is not None else None,
                "title": s.title,
                "positive_group_id": s.positive_group_id,
                "positive_video_ids": s.positive_video_ids,
                "label_source": s.label_source,
                "split": s.split,
                "modality_availability": s.modality_availability,
                "feature_dim": s.feature_dim,
                "metadata": s.metadata,
            })
        return records

    @classmethod
    def from_dict_list(cls, records: List[Dict[str, Any]], feature_dim: int = DEFAULT_FEATURE_DIM) -> "MultimodalTrainingDataset":
        """Reconstruct dataset from list of records."""
        samples = []
        for r in records:
            sample = MultimodalTrainingSample(
                video_id=r["video_id"],
                visual_embedding=torch.tensor(r["visual_embedding"], dtype=torch.float32) if r.get("visual_embedding") is not None else None,
                audio_embedding=torch.tensor(r["audio_embedding"], dtype=torch.float32) if r.get("audio_embedding") is not None else None,
                text_embedding=torch.tensor(r["text_embedding"], dtype=torch.float32) if r.get("text_embedding") is not None else None,
                title=r.get("title"),
                positive_group_id=r.get("positive_group_id"),
                positive_video_ids=r.get("positive_video_ids", []),
                label_source=r.get("label_source", "curated"),
                split=r.get("split", "train"),
                modality_availability=r.get("modality_availability", [False, False, False]),
                feature_dim=r.get("feature_dim", feature_dim),
                metadata=r.get("metadata", {}),
            )
            samples.append(sample)
        return cls(samples=samples, feature_dim=feature_dim)
