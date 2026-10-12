"""
Build Multimodal Features Script (Phase 12B).

Extracts and persists real visual (ResNet-50 frames & thumbnails) and audio (Wav2Vec2)
features from verified media assets.

Usage:
    python scripts/build_multimodal_features.py [--batch-size 10] [--dry-run] [--force]
"""

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sqlalchemy import select, distinct
from src.db.database import get_database
from src.db.models import VideoAsset, AssetType, AssetStatus
from src.services.feature_extraction_service import FeatureExtractionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract and persist multimodal video features.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for extraction processing")
    parser.add_argument("--dry-run", action="store_true", help="Audit eligible videos without writing features")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing features")
    parser.add_argument("--video-id", type=str, default=None, help="Extract for specific video ID only")
    args = parser.parse_args()

    db = get_database()
    feature_service = FeatureExtractionService()

    with db.get_session() as session:
        # Find all distinct video_ids with ready assets
        stmt = select(distinct(VideoAsset.video_id)).where(VideoAsset.status == AssetStatus.READY.value)
        if args.video_id:
            stmt = stmt.where(VideoAsset.video_id == args.video_id)

        video_ids = list(session.execute(stmt).scalars().all())
        logger.info("Found %d distinct videos with ready assets eligible for extraction.", len(video_ids))

        if args.dry_run:
            logger.info("[DRY-RUN] Eligible videos: %s", video_ids)
            print(f"[DRY-RUN] Eligible videos: {len(video_ids)}")
            return

        t0 = time.perf_counter()
        results = feature_service.batch_extract_features(
            video_ids=video_ids,
            session=session,
            batch_size=args.batch_size,
            force=args.force,
        )
        elapsed = time.perf_counter() - t0

        counts = results["counts"]
        logger.info("=== Feature Extraction Run Complete ===")
        logger.info("Total videos processed: %d", counts["total"])
        logger.info("Visual frames extractions (2048-D & 512-D): %d", counts["visual_frames_success"])
        logger.info("Visual thumbnail extractions (2048-D & 512-D): %d", counts["visual_thumbnail_success"])
        logger.info("Audio extractions: %d", counts["audio_success"])
        logger.info("Audio silent waveforms rejected: %d", counts["audio_rejected_silent"])
        logger.info("Failed videos: %d", counts["failed_videos"])
        logger.info("Elapsed time: %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
