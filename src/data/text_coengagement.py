"""
Text-Only Metadata Co-Engagement Dataset and Feature Extractor.

Experiment: EXP_MICROLENS_TEXT_COENGAGEMENT_V1
Extracts genuine chronological co-engagement transitions from user interaction logs,
applies strict user-disjoint train/val/test splitting to prevent user leakage,
and caches frozen BERT representations for efficient projection-head training.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

from src.features.text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class CoengagementPair:
    """A verified user sequential watch transition."""
    user_id: int
    anchor_item: int
    positive_item: int
    delta_seconds: float
    anchor_timestamp: int
    positive_timestamp: int
    split: str = "train"


class TextCoengagementBuilder:
    """
    Constructs genuine sequential co-engagement pairs with user-disjoint splitting.
    """

    def __init__(
        self,
        pairs_csv: str = "data/raw/microlens/MicroLens-50k_pairs.csv",
        titles_csv: str = "data/raw/microlens/MicroLens-50k_titles.csv",
        max_delta_seconds: float = 86400.0,  # 24 hours
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.pairs_csv = Path(pairs_csv)
        self.titles_csv = Path(titles_csv)
        self.max_delta_seconds = max_delta_seconds
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def build_dataset(
        self,
        max_users: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build user-disjoint co-engagement datasets.
        """
        if not self.pairs_csv.exists() or not self.titles_csv.exists():
            raise FileNotFoundError(f"MicroLens files not found: {self.pairs_csv}, {self.titles_csv}")

        # 1. Load titles
        titles_df = pd.read_csv(self.titles_csv)
        titles_map = dict(zip(titles_df["item"].astype(int), titles_df["title"].astype(str)))

        # 2. Load pairs and sort chronologically by user
        pairs_df = pd.read_csv(self.pairs_csv)
        pairs_df["user"] = pairs_df["user"].astype(int)
        pairs_df["item"] = pairs_df["item"].astype(int)
        pairs_df["timestamp"] = pairs_df["timestamp"].astype(int)
        pairs_df.sort_values(by=["user", "timestamp"], inplace=True)

        # 3. Extract consecutive transitions
        pairs_df["prev_user"] = pairs_df["user"].shift(1)
        pairs_df["prev_timestamp"] = pairs_df["timestamp"].shift(1)
        pairs_df["prev_item"] = pairs_df["item"].shift(1)

        same_user = pairs_df["user"] == pairs_df["prev_user"]
        delta_s = (pairs_df["timestamp"] - pairs_df["prev_timestamp"]) / 1000.0
        within_window = delta_s <= self.max_delta_seconds
        valid_transitions = same_user & within_window & (delta_s >= 0)

        valid_df = pairs_df[valid_transitions].copy()
        valid_df["delta_seconds"] = delta_s[valid_transitions]

        # 4. Determine eligible users
        eligible_users = sorted(valid_df["user"].unique().tolist())
        total_eligible = len(eligible_users)

        # Shuffle users deterministically
        rng = random.Random(self.seed)
        shuffled_users = list(eligible_users)
        rng.shuffle(shuffled_users)

        if max_users and max_users < total_eligible:
            selected_users = shuffled_users[:max_users]
        else:
            selected_users = shuffled_users

        n_selected = len(selected_users)
        n_train = int(n_selected * self.train_ratio)
        n_val = int(n_selected * self.val_ratio)

        train_users = set(selected_users[:n_train])
        val_users = set(selected_users[n_train:n_train + n_val])
        test_users = set(selected_users[n_train + n_val:])

        # 5. Partition transitions
        selected_set = set(selected_users)
        filtered_df = valid_df[valid_df["user"].isin(selected_set)]

        train_pairs: List[CoengagementPair] = []
        val_pairs: List[CoengagementPair] = []
        test_pairs: List[CoengagementPair] = []

        all_participating_items: Set[int] = set()

        for _, row in filtered_df.iterrows():
            u = int(row["user"])
            a = int(row["prev_item"])
            p = int(row["item"])

            # Skip self-transitions if any
            if a == p:
                continue

            all_participating_items.add(a)
            all_participating_items.add(p)

            pair_obj = CoengagementPair(
                user_id=u,
                anchor_item=a,
                positive_item=p,
                delta_seconds=float(row["delta_seconds"]),
                anchor_timestamp=int(row["prev_timestamp"]),
                positive_timestamp=int(row["timestamp"]),
            )

            if u in train_users:
                pair_obj.split = "train"
                train_pairs.append(pair_obj)
            elif u in val_users:
                pair_obj.split = "val"
                val_pairs.append(pair_obj)
            elif u in test_users:
                pair_obj.split = "test"
                test_pairs.append(pair_obj)

        stats = {
            "total_users_in_dataset": pairs_df["user"].nunique(),
            "eligible_users_total": total_eligible,
            "selected_users": n_selected,
            "train_users": len(train_users),
            "val_users": len(val_users),
            "test_users": len(test_users),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "unique_participating_items": len(all_participating_items),
        }

        return {
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
            "titles_map": titles_map,
            "unique_items": sorted(all_participating_items),
            "stats": stats,
        }


class BertFeatureExtractor:
    """
    Extracts and caches frozen BERT-base-uncased 768-D features.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        max_length: int = 64,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.max_length = max_length
        self.processor = TextProcessor()

        # Load with local_files_only=True
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = BertModel.from_pretrained(model_name, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        # Ensure backbone is strictly frozen
        for p in self.model.parameters():
            p.requires_grad = False

    def extract_features(
        self,
        item_ids: List[int],
        titles_map: Dict[int, str],
        batch_size: int = 64,
        cache_file: Optional[Path] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract frozen BERT [CLS] representations for items with optional caching.
        """
        if cache_file and cache_file.exists():
            logger.info("Loading cached BERT features from %s", cache_file)
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            if payload.get("model_name") == self.model_name:
                features = payload["features"]
                # Verify all requested items are in cache
                if all(item in features for item in item_ids):
                    return features

        logger.info("Extracting BERT features for %d items...", len(item_ids))
        t0 = time.time()
        features: Dict[int, torch.Tensor] = {}

        for i in range(0, len(item_ids), batch_size):
            batch_items = item_ids[i:i + batch_size]
            cleaned_texts = [
                self.processor.process(titles_map.get(item, ""))[0] or "video"
                for item in batch_items
            ]

            encoded = self.tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] representation (768-D)
                cls_reps = outputs.last_hidden_state[:, 0, :].cpu()

            for item_id, feat in zip(batch_items, cls_reps):
                features[item_id] = feat

        elapsed = time.time() - t0
        logger.info("Feature extraction complete in %.2fs (%.1f items/sec)", elapsed, len(item_ids) / max(elapsed, 0.001))

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_name": self.model_name,
                "dimension": 768,
                "features": features,
                "timestamp": time.time(),
            }
            torch.save(payload, cache_file)
            logger.info("Saved feature cache to %s", cache_file)

        return features


class PairedTextFeatureDataset(Dataset):
    """
    Yields (anchor_feature, positive_feature, anchor_item, positive_item).
    """

    def __init__(
        self,
        pairs: List[CoengagementPair],
        features: Dict[int, torch.Tensor],
    ):
        self.pairs = pairs
        self.features = features

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        a_feat = self.features[p.anchor_item]
        p_feat = self.features[p.positive_item]

        return {
            "anchor_feature": a_feat,
            "positive_feature": p_feat,
            "anchor_item": p.anchor_item,
            "positive_item": p.positive_item,
            "user_id": p.user_id,
        }
