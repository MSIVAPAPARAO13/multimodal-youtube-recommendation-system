"""
Synthetic Visual Evaluation Script (Phase 6).

Executes an offline, controlled visual evaluation on deterministic synthetic image/video bundles
across 5 semantic categories:
- programming
- cybersecurity
- machine_learning
- web_development
- cloud_computing

Measures:
1. Representation Quality Sanity Checks:
   - Embedding dimension (512-D)
   - Unit hypersphere L2 normalization
   - Deterministic consistency across repeated runs
   - Absence of NaN/Inf values
2. Information Retrieval Metrics:
   - Precision@1, Precision@3
   - Recall@3
   - Mean Reciprocal Rank (MRR)
   - Mean Average Precision (MAP)

NOTE: This benchmark is an offline pipeline/representation sanity evaluation.
It does NOT claim to represent real-world YouTube recommendation performance.
"""

import json
from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.features.visual_encoder import ResNet50VisualEncoder
from src.services.multimodal_inference import VisualEmbeddingService

CATEGORIES = [
    "programming",
    "cybersecurity",
    "machine_learning",
    "web_development",
    "cloud_computing",
]


def generate_synthetic_pattern(category: str, variant: int, width: int = 224, height: int = 224) -> Image.Image:
    """Generate deterministic geometric patterns characterizing each synthetic category."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if category == "programming":
        # Blue gradient with horizontal code-like lines
        base_color = (20, 40 + variant * 10, 150 + variant * 20)
        img.paste(base_color, [0, 0, width, height])
        for y in range(20, height - 20, 20):
            line_len = 50 + ((y * 7 + variant * 30) % (width - 80))
            draw.line([(30, y), (30 + line_len, y)], fill=(180, 220, 255), width=4)

    elif category == "cybersecurity":
        # Dark green / black matrix-like grid
        base_color = (5, 20 + variant * 5, 10)
        img.paste(base_color, [0, 0, width, height])
        for x in range(15, width, 25):
            for y in range(15, height, 25):
                draw.rectangle([x, y, x + 8, y + 8], fill=(0, 220 - variant * 20, 80))

    elif category == "machine_learning":
        # Purple / magenta neural network nodes
        base_color = (60 + variant * 10, 10, 80 + variant * 15)
        img.paste(base_color, [0, 0, width, height])
        nodes = [(40, 60), (40, 160), (112, 112), (180, 60), (180, 160)]
        for i, (x1, y1) in enumerate(nodes):
            for j, (x2, y2) in enumerate(nodes):
                if i < j:
                    draw.line([(x1, y1), (x2, y2)], fill=(200, 100, 220), width=2)
        for x, y in nodes:
            draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(255, 180, 255))

    elif category == "web_development":
        # Warm orange / layout blocks
        base_color = (180 + variant * 15, 80 + variant * 10, 20)
        img.paste(base_color, [0, 0, width, height])
        draw.rectangle([20, 20, width - 20, 50], fill=(255, 200, 120))  # Navbar
        draw.rectangle([20, 60, 80, height - 20], fill=(240, 140, 60))  # Sidebar
        draw.rectangle([90, 60, width - 20, height - 20], fill=(255, 240, 200))  # Body

    elif category == "cloud_computing":
        # Sky cyan / concentric circles
        base_color = (10, 120 + variant * 15, 180 + variant * 10)
        img.paste(base_color, [0, 0, width, height])
        for r in range(20, 90, 20):
            draw.ellipse([112 - r, 112 - r, 112 + r, 112 + r], outline=(220, 245, 255), width=3)

    return img


def create_synthetic_dataset(output_dir: Path, items_per_category: int = 3) -> list[dict]:
    """Create local synthetic video and thumbnail dataset for visual evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = []

    for cat in CATEGORIES:
        for var in range(items_per_category):
            item_id = f"{cat}_{var}"
            item_dir = output_dir / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            vid_path = item_dir / "video.mp4"
            thumb_path = item_dir / "thumbnail.jpg"

            # Create thumbnail
            thumb_img = generate_synthetic_pattern(cat, var, width=320, height=180)
            thumb_img.save(thumb_path)

            # Create video (8 frames showing pattern with slight variations)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (224, 224))
            for f_idx in range(8):
                frame_img = generate_synthetic_pattern(cat, (var + f_idx) % 5, width=224, height=224)
                frame_bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()

            dataset.append({
                "id": item_id,
                "category": cat,
                "variant": var,
                "video_path": str(vid_path),
                "thumbnail_path": str(thumb_path),
            })

    return dataset


def evaluate():
    print("=" * 65)
    print("  SYNTHETIC VISUAL EVALUATION BENCHMARK (PHASE 6)")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"[*] Initializing VisualEmbeddingService on {device}...")
    encoder = ResNet50VisualEncoder(feature_dim=512, pretrained=True, freeze_backbone=True).to(device)
    encoder.eval()

    visual_service = VisualEmbeddingService(video_encoder=encoder, device=device)

    fixtures_dir = Path("data/fixtures/synthetic_visual_eval")
    print(f"[*] Generating synthetic visual bundles in {fixtures_dir}...")
    dataset = create_synthetic_dataset(fixtures_dir, items_per_category=3)
    print(f"[*] Created {len(dataset)} items across {len(CATEGORIES)} categories.\n")

    # 1. Representation Sanity Checks
    embeddings = []
    labels = []
    ids = []

    norm_checks = []
    nan_checks = []
    dim_checks = []
    deterministic_checks = []

    for item in dataset:
        res1 = visual_service.embed_video(video_path=item["video_path"], thumbnail_path=item["thumbnail_path"])
        res2 = visual_service.embed_video(video_path=item["video_path"], thumbnail_path=item["thumbnail_path"])

        emb1 = res1.embedding
        emb2 = res2.embedding

        # Sanity 1: Shape is (1, 512)
        dim_checks.append(emb1.shape == (1, 512))

        # Sanity 2: L2 Normalized
        norm = torch.norm(emb1, p=2).item()
        norm_checks.append(abs(norm - 1.0) < 1e-4)

        # Sanity 3: No NaN/Inf
        nan_checks.append(not torch.isnan(emb1).any().item() and not torch.isinf(emb1).any().item())

        # Sanity 4: Deterministic
        sim_repeat = F.cosine_similarity(emb1, emb2).item()
        deterministic_checks.append(abs(sim_repeat - 1.0) < 1e-4)

        embeddings.append(emb1)
        labels.append(item["category"])
        ids.append(item["id"])

    emb_matrix = torch.cat(embeddings, dim=0)  # (N, 512)

    # 2. Nearest-Neighbor Retrieval Evaluation
    N = len(dataset)
    sim_matrix = torch.mm(emb_matrix, emb_matrix.t()).cpu().numpy()  # (N, N) cosine similarity

    p_at_1_list = []
    p_at_3_list = []
    r_at_3_list = []
    mrr_list = []
    ap_list = []

    for i in range(N):
        query_cat = labels[i]
        # Rank all items except query itself
        scores = [(sim_matrix[i][j], labels[j]) for j in range(N) if j != i]
        scores.sort(key=lambda x: x[0], reverse=True)

        # Precision@1
        p_at_1 = 1.0 if scores[0][1] == query_cat else 0.0
        p_at_1_list.append(p_at_1)

        # Precision@3 & Recall@3 (ground truth relevant = 2 other items in category)
        top3_matches = sum(1 for _, cat in scores[:3] if cat == query_cat)
        p_at_3_list.append(top3_matches / 3.0)
        r_at_3_list.append(top3_matches / 2.0)

        # MRR (first relevant item position)
        rank_first_rel = None
        for rank, (_, cat) in enumerate(scores, 1):
            if cat == query_cat:
                rank_first_rel = rank
                break
        mrr_list.append(1.0 / rank_first_rel if rank_first_rel else 0.0)

        # Average Precision (AP)
        num_rel = 0
        prec_sum = 0.0
        for rank, (_, cat) in enumerate(scores, 1):
            if cat == query_cat:
                num_rel += 1
                prec_sum += num_rel / rank
        ap_list.append(prec_sum / 2.0 if num_rel > 0 else 0.0)

    p1_mean = float(np.mean(p_at_1_list))
    p3_mean = float(np.mean(p_at_3_list))
    r3_mean = float(np.mean(r_at_3_list))
    mrr_mean = float(np.mean(mrr_list))
    map_mean = float(np.mean(ap_list))

    print("=" * 65)
    print("  REPRESENTATION QUALITY SANITY CHECKS")
    print("=" * 65)
    print(f"  [+] 512-D Embedding Dimension:   {all(dim_checks)} ({sum(dim_checks)}/{len(dim_checks)})")
    print(f"  [+] L2 Hypersphere Normalization: {all(norm_checks)} ({sum(norm_checks)}/{len(norm_checks)})")
    print(f"  [+] Deterministic Repeatability:  {all(deterministic_checks)} ({sum(deterministic_checks)}/{len(deterministic_checks)})")
    print(f"  [+] NaN / Inf Immunity:           {all(nan_checks)} ({sum(nan_checks)}/{len(nan_checks)})")

    print("\n" + "=" * 65)
    print("  INFORMATION RETRIEVAL EVALUATION RESULTS")
    print(f"  Dataset: {N} synthetic videos/thumbnails across {len(CATEGORIES)} categories")
    print("=" * 65)
    print(f"  Precision@1: {p1_mean:8.4f} ({p1_mean * 100:.1f}%)")
    print(f"  Precision@3: {p3_mean:8.4f} ({p3_mean * 100:.1f}%)")
    print(f"  Recall@3:    {r3_mean:8.4f} ({r3_mean * 100:.1f}%)")
    print(f"  MRR:         {mrr_mean:8.4f}")
    print(f"  MAP:         {map_mean:8.4f}")
    print("=" * 65)
    print("\n[NOTE] These metrics benchmark mathematical consistency and visual ranking")
    print("pipeline functionality on synthetic patterns. They do NOT claim real-world")
    print("YouTube recommendation accuracy.")
    print("=" * 65 + "\n")

    results_dict = {
        "dataset_size": N,
        "categories": CATEGORIES,
        "sanity_checks": {
            "dim_512_passed": all(dim_checks),
            "l2_norm_passed": all(norm_checks),
            "deterministic_passed": all(deterministic_checks),
            "no_nan_passed": all(nan_checks),
        },
        "retrieval_metrics": {
            "precision_at_1": round(p1_mean, 4),
            "precision_at_3": round(p3_mean, 4),
            "recall_at_3": round(r3_mean, 4),
            "mrr": round(mrr_mean, 4),
            "map": round(map_mean, 4),
        },
    }

    out_json = fixtures_dir / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[*] Evaluation results written to {out_json}")
    return results_dict


if __name__ == "__main__":
    evaluate()
