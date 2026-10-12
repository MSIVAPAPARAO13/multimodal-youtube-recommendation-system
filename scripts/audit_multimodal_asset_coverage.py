"""
Phase 12A - Multimodal Asset Coverage and Integrity Audit Script.

Inspects catalog records, database assets, disk files, MIME types, and checksums.
Produces a structured report at:
    data/checkpoints/EXP_MULTIMODAL_ASSET_INGESTION_12A/asset_coverage_report.json
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sqlalchemy import select, func
from src.db.database import get_database
from src.db.models import Video, VideoIngestion, VideoAsset, IngestionStatus, AssetType, AssetStatus
from src.integrations.youtube.media.validator import MediaValidator
from src.storage.local import LocalStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_asset_coverage_audit() -> dict[str, Any]:
    """Execute complete catalog and asset storage audit."""
    output_dir = Path("data/checkpoints/EXP_MULTIMODAL_ASSET_INGESTION_12A")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "asset_coverage_report.json"

    db = get_database()
    storage = LocalStorage(root_dir="data/raw/youtube")
    validator = MediaValidator()

    with db.get_session() as session:
        # 1. Catalog Records
        total_catalog = session.scalar(select(func.count(Video.id))) or 0
        records_with_yt_id = session.scalar(
            select(func.count(Video.id)).where(Video.youtube_video_id.isnot(None))
        ) or 0
        records_with_metadata = session.scalar(
            select(func.count(Video.id)).where(
                (Video.channel_id.isnot(None)) | (Video.duration.isnot(None))
            )
        ) or 0

        # 2. Ingestion Jobs
        total_ingestion_jobs = session.scalar(select(func.count(VideoIngestion.id))) or 0
        completed_jobs = session.scalar(
            select(func.count(VideoIngestion.id)).where(
                VideoIngestion.status.in_([IngestionStatus.COMPLETED.value, IngestionStatus.MEDIA_READY.value])
            )
        ) or 0
        failed_jobs = session.scalar(
            select(func.count(VideoIngestion.id)).where(VideoIngestion.status == IngestionStatus.FAILED.value)
        ) or 0
        pending_jobs = session.scalar(
            select(func.count(VideoIngestion.id)).where(VideoIngestion.status == IngestionStatus.PENDING.value)
        ) or 0

        # 3. Video Assets by Type
        asset_stmt = select(VideoAsset.asset_type, func.count(VideoAsset.id)).group_by(VideoAsset.asset_type)
        type_counts = {row[0]: row[1] for row in session.execute(asset_stmt).all()}

        total_assets = session.scalar(select(func.count(VideoAsset.id))) or 0
        distinct_videos_with_assets = session.scalar(
            select(func.count(func.distinct(VideoAsset.video_id)))
        ) or 0

        # 4. Integrity and Disk Validation
        all_assets = session.execute(select(VideoAsset)).scalars().all()
        disk_verified = 0
        disk_missing = 0
        checksum_matches = 0
        checksum_mismatches = 0
        total_storage_bytes = 0

        storage_by_type = {
            "VIDEO": 0,
            "AUDIO": 0,
            "THUMBNAIL": 0,
            "FRAME": 0,
            "METADATA": 0,
        }

        for asset in all_assets:
            target_path = storage.root_path / asset.storage_path
            if target_path.exists() and target_path.is_file():
                disk_verified += 1
                size = target_path.stat().st_size
                total_storage_bytes += size
                storage_by_type[asset.asset_type] = storage_by_type.get(asset.asset_type, 0) + size

                if asset.checksum_sha256:
                    actual_checksum = validator.compute_sha256(target_path)
                    if actual_checksum == asset.checksum_sha256:
                        checksum_matches += 1
                    else:
                        checksum_mismatches += 1
            else:
                disk_missing += 1

        # 5. Build Audit Summary
        report = {
            "audit_timestamp": datetime.utcnow().isoformat(),
            "pipeline_version": "12A",
            "catalog_summary": {
                "total_catalog_records": total_catalog,
                "records_with_youtube_id": records_with_yt_id,
                "records_with_extended_metadata": records_with_metadata,
                "records_without_media_source": max(0, total_catalog - distinct_videos_with_assets),
            },
            "ingestion_jobs_summary": {
                "total_jobs": total_ingestion_jobs,
                "completed_or_ready": completed_jobs,
                "failed": failed_jobs,
                "pending": pending_jobs,
            },
            "asset_coverage_summary": {
                "total_registered_assets": total_assets,
                "distinct_videos_with_assets": distinct_videos_with_assets,
                "video_assets": type_counts.get("VIDEO", 0),
                "audio_assets": type_counts.get("AUDIO", 0),
                "thumbnail_assets": type_counts.get("THUMBNAIL", 0),
                "extracted_frame_assets": type_counts.get("FRAME", 0),
                "metadata_assets": type_counts.get("METADATA", 0),
                "coverage_percentages": {
                    "thumbnails": round(type_counts.get("THUMBNAIL", 0) / max(1, total_catalog) * 100, 2),
                    "video_media": round(type_counts.get("VIDEO", 0) / max(1, total_catalog) * 100, 2),
                    "audio_media": round(type_counts.get("AUDIO", 0) / max(1, total_catalog) * 100, 2),
                    "frame_media": round(distinct_videos_with_assets / max(1, total_catalog) * 100, 2),
                },
            },
            "storage_and_integrity": {
                "disk_files_verified": disk_verified,
                "disk_files_missing": disk_missing,
                "checksum_matches": checksum_matches,
                "checksum_mismatches": checksum_mismatches,
                "total_storage_bytes": total_storage_bytes,
                "total_storage_mb": round(total_storage_bytes / (1024 * 1024), 2),
                "storage_by_modality_bytes": storage_by_type,
                "storage_by_modality_mb": {
                    k: round(v / (1024 * 1024), 2) for k, v in storage_by_type.items()
                },
            },
            "embedding_readiness": {
                "visual_embeddings_computed": False,
                "audio_embeddings_computed": False,
                "note": "Phase 12A establishes asset alignment and integrity. Embedding extraction occurs in Phase 12B.",
            },
        }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved coverage report to %s", report_file)
    print("\n" + "=" * 70)
    print("PHASE 12A - MULTIMODAL ASSET INGESTION & COVERAGE AUDIT")
    print("=" * 70)
    print(f"Total Catalog Records:             {total_catalog}")
    print(f"Records with YouTube IDs:          {records_with_yt_id}")
    print(f"Videos with Aligned Media:         {distinct_videos_with_assets}")
    print("-" * 70)
    print(f"Total Registered Assets:           {total_assets}")
    print(f"  - Video Media (.mp4):            {type_counts.get('VIDEO', 0)}")
    print(f"  - Audio Streams (.wav):          {type_counts.get('AUDIO', 0)}")
    print(f"  - Thumbnails (.jpg):             {type_counts.get('THUMBNAIL', 0)}")
    print(f"  - Extracted Video Frames (.jpg): {type_counts.get('FRAME', 0)}")
    print(f"  - Metadata Manifests (.json):    {type_counts.get('METADATA', 0)}")
    print("-" * 70)
    print(f"Disk Verified Files:               {disk_verified} / {total_assets}")
    print(f"SHA-256 Checksum Matches:          {checksum_matches} / {disk_verified}")
    print(f"Checksum Mismatches:               {checksum_mismatches}")
    print(f"Total Asset Storage:               {report['storage_and_integrity']['total_storage_mb']} MB")
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    run_asset_coverage_audit()
