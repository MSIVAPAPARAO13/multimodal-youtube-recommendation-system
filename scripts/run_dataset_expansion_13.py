"""
Phase 13 - Large-Scale Media Ingestion & Dataset Expansion CLI Script.
Executes batch media acquisition and audio recovery using AssetIngestionOrchestrator.
Supports resumability, rate limiting, bounded concurrency, storage budget checks, and deduplication.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, List

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from src.services.youtube_ingestion import YouTubeIngestionService, AssetIngestionOrchestrator
from src.db.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_dataset_expansion_13")


def load_candidate_items(target_count: int = 1000) -> List[dict]:
    """
    Load candidate items from MicroLens titles CSV prioritizing available local media.
    """
    candidates = []
    seen = set()

    # 1. First priority: any physical videos present in data/raw/microlens/videos/
    local_vid_dir = Path("data/raw/microlens/videos")
    if local_vid_dir.exists():
        for mp4_path in sorted(local_vid_dir.glob("*.mp4")):
            vid = mp4_path.stem
            seen.add(vid)
            candidates.append({
                "video_id": vid,
                "local_video_path": str(mp4_path),
                "title": f"MicroLens Local Video {vid}",
            })

    # 2. Second priority: catalog entries from MicroLens-50k_titles.csv
    titles_csv = Path("data/raw/microlens/MicroLens-50k_titles.csv")
    if titles_csv.exists():
        df = pd.read_csv(titles_csv)
        for _, row in df.iterrows():
            vid = str(row["item"])
            if vid not in seen:
                seen.add(vid)
                candidates.append({
                    "video_id": vid,
                    "title": str(row["title"]),
                })
            if len(candidates) >= target_count:
                break

    return candidates[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Batch Ingestion Pipeline")
    parser.add_argument("--target", type=int, default=1000, help="Target number of candidate videos")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch submission size")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency worker threads (1-8)")
    parser.add_argument("--max-storage-gb", type=float, default=10.0, help="Maximum storage budget in GB")
    parser.add_argument("--rate-limit", type=float, default=0.0, help="Rate limit delay in seconds between items")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion of already completed videos")
    parser.add_argument("--resume-job", type=str, default=None, help="Resume an existing job ID from checkpoint")
    args = parser.parse_args()

    settings = get_settings()
    service = YouTubeIngestionService(settings=settings)
    orchestrator = AssetIngestionOrchestrator(
        ingestion_service=service,
        max_concurrency=args.concurrency,
        batch_size=args.batch_size,
        rate_limit_delay_seconds=args.rate_limit,
        max_storage_gb=args.max_storage_gb,
    )

    if args.resume_job:
        logger.info("Resuming job %s...", args.resume_job)
        job_id = orchestrator.resume_job(args.resume_job)
    else:
        logger.info("Loading up to %d candidates for ingestion...", args.target)
        candidates = load_candidate_items(target_count=args.target)
        logger.info("Submitting batch of %d candidates...", len(candidates))
        job_id = orchestrator.submit_batch(candidates, force=args.force)

    logger.info("Batch job submitted: %s", job_id)

    # Monitor until completion
    while True:
        status = orchestrator.get_job_status(job_id)
        if not status:
            logger.warning("Job status unavailable.")
            break

        st = status.get("status")
        comp = status.get("completed", 0)
        skip = status.get("skipped", 0)
        fail = status.get("failed", 0)
        tot = status.get("total", 0)

        logger.info("Job %s [%s]: %d/%d (completed=%d, skipped=%d, failed=%d)", job_id, st, comp + skip + fail, tot, comp, skip, fail)

        if st in ["COMPLETED", "PARTIAL_FAILURE"]:
            break
        time.sleep(1.0)

    final_status = orchestrator.get_job_status(job_id)
    print("\n================ BATCH INGESTION SUMMARY ================")
    print(f"Job ID: {job_id}")
    print(f"Final Status: {final_status.get('status')}")
    print(f"Total Items: {final_status.get('total')}")
    print(f"Completed: {final_status.get('completed')}")
    print(f"Skipped (Deduplicated): {final_status.get('skipped')}")
    print(f"Failed: {final_status.get('failed')}")
    print(f"Errors Logged: {len(final_status.get('errors', []))}")
    print("=========================================================")


if __name__ == "__main__":
    main()
