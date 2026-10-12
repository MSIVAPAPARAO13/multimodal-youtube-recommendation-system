"""
Benchmark Script for Multimodal Inference Pipeline.

Measures:
1. Model initialization time
2. Video encoding time (ResNet-50)
3. Audio encoding time (Wav2Vec2)
4. Text encoding time (BERT)
5. Fusion time (GatedFusion + L2 Normalization)
6. Total end-to-end inference time
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
from src.db.config import get_settings
from src.features.video_encoder import VideoEncoder
from src.features.audio_encoder import AudioEncoder
from src.features.text_encoder import TextEncoder
from src.models.fusion import GatedFusion
from src.services.multimodal_inference import MultimodalInferenceService


def benchmark():
    settings = get_settings()
    device = torch.device("cuda" if (torch.cuda.is_available() and settings.DEVICE in ("auto", "cuda")) else "cpu")
    print("=" * 60)
    print("  MULTIMODAL INFERENCE BENCHMARK")
    print(f"  Device: {device}")
    print(f"  Torch:  {torch.__version__}")
    print("=" * 60)

    # 1. Model Initialization Benchmark
    t0 = time.perf_counter()
    video_encoder = VideoEncoder(backbone=settings.RESNET_BACKBONE, feature_dim=512, pretrained=True, freeze_backbone=True).to(device)
    video_encoder.eval()
    t_video_init = (time.perf_counter() - t0) * 1000

    print(f"[*] ResNet-50 VideoEncoder init:  {t_video_init:8.2f} ms")

    # Use pretrained=False or config for quick audio/text init in offline benchmark
    t0 = time.perf_counter()
    audio_encoder = AudioEncoder(model_name=settings.WAV2VEC2_MODEL, feature_dim=512, freeze_backbone=True, pretrained=False).to(device)
    audio_encoder.eval()
    t_audio_init = (time.perf_counter() - t0) * 1000
    print(f"[*] Wav2Vec2 AudioEncoder init:  {t_audio_init:8.2f} ms")

    t0 = time.perf_counter()
    text_encoder = TextEncoder(model_name=settings.BERT_MODEL, feature_dim=512, freeze_backbone=True, pretrained=False).to(device)
    text_encoder.eval()
    t_text_init = (time.perf_counter() - t0) * 1000
    print(f"[*] BERT TextEncoder init:       {t_text_init:8.2f} ms")

    t0 = time.perf_counter()
    fusion = GatedFusion(input_dim=512, output_dim=512).to(device)
    fusion.eval()
    t_fusion_init = (time.perf_counter() - t0) * 1000
    print(f"[*] GatedFusion init:            {t_fusion_init:8.2f} ms")

    total_init_time = t_video_init + t_audio_init + t_text_init + t_fusion_init
    print(f"[*] Total Model Init Time:       {total_init_time:8.2f} ms")
    print("-" * 60)

    # Prepare inputs
    frames = torch.randn(1, 8, 3, 224, 224, device=device)
    audio = torch.randn(1, 80000, device=device)
    text_input = "Learn Machine Learning: Deep Multimodal Recommendations"

    # Warmup
    with torch.no_grad():
        _ = video_encoder(frames)
        _ = audio_encoder(audio)
        _ = text_encoder(texts=[text_input])
        v_dummy = torch.randn(1, 512, device=device)
        a_dummy = torch.randn(1, 512, device=device)
        t_dummy = torch.randn(1, 512, device=device)
        _ = fusion(v_dummy, a_dummy, t_dummy)

    # Benchmark Iterations
    num_runs = 10
    v_times, a_times, t_times, f_times, total_times = [], [], [], [], []

    print(f"Running {num_runs} benchmark iterations...")

    with torch.no_grad():
        for _ in range(num_runs):
            # Video
            t_start = time.perf_counter()
            v_emb = video_encoder(frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_v = (time.perf_counter() - t_start) * 1000
            v_times.append(t_v)

            # Audio
            t_start = time.perf_counter()
            a_emb = audio_encoder(audio)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_a = (time.perf_counter() - t_start) * 1000
            a_times.append(t_a)

            # Text
            t_start = time.perf_counter()
            t_emb = text_encoder(texts=[text_input])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_t = (time.perf_counter() - t_start) * 1000
            t_times.append(t_t)

            # Fusion + L2 Norm
            t_start = time.perf_counter()
            fused = fusion(v_emb, a_emb, t_emb)
            norm = torch.nn.functional.normalize(fused, p=2, dim=-1)
            _ = torch.nan_to_num(norm, nan=0.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_f = (time.perf_counter() - t_start) * 1000
            f_times.append(t_f)

            total_times.append(t_v + t_a + t_t + t_f)

    avg_v = sum(v_times) / num_runs
    avg_a = sum(a_times) / num_runs
    avg_t = sum(t_times) / num_runs
    avg_f = sum(f_times) / num_runs
    avg_total = sum(total_times) / num_runs

    print("=" * 60)
    print("  MULTIMODAL BENCHMARK RESULTS (Average over 10 runs)")
    print("=" * 60)
    print(f"  Video Encoding (ResNet-50, 8 frames):   {avg_v:8.2f} ms")
    print(f"  Audio Encoding (Wav2Vec2, 5s waveform):  {avg_a:8.2f} ms")
    print(f"  Text Encoding (BERT, title string):      {avg_t:8.2f} ms")
    print(f"  Multimodal Fusion (GatedFusion + L2):    {avg_f:8.2f} ms")
    print(f"  --------------------------------------------------")
    print(f"  Total Multimodal Inference Latency:      {avg_total:8.2f} ms")
    print(f"  Theoretical Throughput (1 thread):       {1000.0 / avg_total:8.2f} req/s")
    print("=" * 60)

    # ----------------------------------------------------
    # PHASE 5: TEXT NLP PIPELINE DETAILED PROFILING
    # ----------------------------------------------------
    text_metrics = benchmark_text_pipeline(text_encoder, device=device, num_runs=num_runs)

    # ----------------------------------------------------
    # PHASE 6: VISUAL & THUMBNAIL PIPELINE DETAILED PROFILING
    # ----------------------------------------------------
    visual_metrics = benchmark_visual_pipeline(device=device, num_runs=num_runs)

    return {
        "video_init_ms": t_video_init,
        "audio_init_ms": t_audio_init,
        "text_init_ms": t_text_init,
        "fusion_init_ms": t_fusion_init,
        "total_init_ms": total_init_time,
        "video_encode_ms": avg_v,
        "audio_encode_ms": avg_a,
        "text_encode_ms": avg_t,
        "fusion_ms": avg_f,
        "total_inference_ms": avg_total,
        **text_metrics,
        **visual_metrics,
    }


def benchmark_text_pipeline(text_encoder: TextEncoder, device: torch.device, num_runs: int = 10) -> dict[str, float]:
    """
    Profile individual components of the YouTube semantic text pipeline:
    1. Preprocessing time (normalization, artifact cleanup, section formatting)
    2. Tokenization time
    3. Chunking overhead
    4. BERT forward inference time
    5. Total end-to-end text embedding latency
    """
    from src.features.text_processor import YouTubeTextInput, TextProcessor
    from src.services.multimodal_inference import TextEmbeddingService

    text_service = TextEmbeddingService(text_encoder=text_encoder, device=device, feature_dim=512)
    processor = TextProcessor()

    sample_input = {
        "title": "Deep Learning: Convolutional and Transformer Architectures",
        "description": "Comprehensive guide on ResNet, BERT, and Wav2Vec2 architectures. In this lecture we explore modern multimodal systems.",
        "tags": ["deep-learning", "pytorch", "transformers", "multimodal"],
        "transcript": (
            "00:00:01.000 --> 00:00:04.000\n[00:01] Hello everyone and welcome to this lecture. [Music]\n"
            "Today we will dissect how deep convolutional layers capture spatial hierarchies. "
            "Then we transition to self-attention mechanisms and query key value representations. "
            "Finally we look at cross-modal gating and fusion networks. "
            * 8  # Produce a multi-chunk long transcript
        )
    }

    # Warmup
    _ = text_service.encode(sample_input)

    t_pre_list = []
    t_tok_list = []
    t_chunk_list = []
    t_bert_list = []
    t_total_list = []

    for _ in range(num_runs):
        # 1. Preprocessing time
        t0 = time.perf_counter()
        structured_text, diag, canonical = processor.process(sample_input)
        t_pre = (time.perf_counter() - t0) * 1000
        t_pre_list.append(t_pre)

        # 2. Tokenization time
        t0 = time.perf_counter()
        raw_tokens = text_encoder.tokenizer(
            structured_text,
            add_special_tokens=False,
            truncation=False,
            return_tensors=None
        )
        t_tok = (time.perf_counter() - t0) * 1000
        t_tok_list.append(t_tok)

        # 3. Chunking overhead
        t0 = time.perf_counter()
        chunk_data = text_encoder.chunker.chunk_text(
            structured_text,
            tokenizer=text_encoder.tokenizer,
            add_special_tokens=True
        )
        t_chunk = (time.perf_counter() - t0) * 1000
        t_chunk_list.append(t_chunk)

        # 4. BERT forward inference
        input_ids = chunk_data["input_ids"].to(device)
        attention_mask = chunk_data["attention_mask"].to(device)
        t0 = time.perf_counter()
        chunk_embs = text_encoder._forward_tokens(input_ids, attention_mask)
        doc_emb = text_encoder.chunker.pool_chunk_embeddings(chunk_embs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_bert = (time.perf_counter() - t0) * 1000
        t_bert_list.append(t_bert)

        # 5. Total end-to-end TextEmbeddingService time
        t0 = time.perf_counter()
        _ = text_service.encode(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_tot = (time.perf_counter() - t0) * 1000
        t_total_list.append(t_tot)

    avg_pre = sum(t_pre_list) / num_runs
    avg_tok = sum(t_tok_list) / num_runs
    avg_chunk = sum(t_chunk_list) / num_runs
    avg_bert = sum(t_bert_list) / num_runs
    avg_tot = sum(t_total_list) / num_runs

    print("\n" + "=" * 60)
    print("  PHASE 5 — TEXT NLP PIPELINE LATENCY BREAKDOWN")
    print(f"  Evaluated on multi-chunk structured YouTube input ({diag.chunk_count} chunks)")
    print("=" * 60)
    print(f"  1. Text Preprocessing (clean, regex, NFKC): {avg_pre:8.2f} ms")
    print(f"  2. WordPiece Tokenization:                 {avg_tok:8.2f} ms")
    print(f"  3. TextChunker Window Overhead:             {avg_chunk:8.2f} ms")
    print(f"  4. BERT Forward Inference + Projection:    {avg_bert:8.2f} ms")
    print(f"  --------------------------------------------------")
    print(f"  Total End-to-End Text Latency:             {avg_tot:8.2f} ms")
    print(f"  Theoretical Text Throughput:               {1000.0 / avg_tot:8.2f} req/s")
    print("=" * 60 + "\n")

    return {
        "text_preprocessing_ms": avg_pre,
        "text_tokenization_ms": avg_tok,
        "text_chunking_overhead_ms": avg_chunk,
        "text_bert_inference_ms": avg_bert,
        "total_text_latency_ms": avg_tot,
    }


def benchmark_visual_pipeline(device: Optional[torch.device] = None, num_runs: int = 10) -> dict[str, float]:
    """
    Profile individual components of the Phase 6 Visual & Thumbnail Pipeline:
    1. Frame sampling latency (FrameSampler with OpenCV)
    2. Image preprocessing latency (VisualProcessor ImageNet 224x224)
    3. ResNet-50 frame inference latency (2048-D backbone + 512-D projection)
    4. Frame aggregation latency (mean pooling)
    5. Thumbnail inference latency (ThumbnailEncoder)
    6. Video visual embedding total latency
    7. Video + thumbnail visual fusion latency
    """
    import tempfile
    import numpy as np
    from PIL import Image
    import cv2
    from src.features.frame_sampler import FrameSampler
    from src.features.visual_processor import VisualProcessor
    from src.features.visual_encoder import ResNet50VisualEncoder
    from src.features.thumbnail_encoder import ThumbnailEncoder
    from src.services.multimodal_inference import VisualEmbeddingService

    device = device or torch.device("cpu")
    sampler = FrameSampler(default_num_frames=8)
    processor = VisualProcessor(target_size=(224, 224), normalize=True)
    encoder = ResNet50VisualEncoder(feature_dim=512, pretrained=True, freeze_backbone=True).to(device)
    encoder.eval()
    thumb_encoder = ThumbnailEncoder(encoder=encoder, processor=processor, device=device, feature_dim=512)
    service = VisualEmbeddingService(
        video_encoder=encoder,
        processor=processor,
        frame_sampler=sampler,
        thumbnail_encoder=thumb_encoder,
        device=device,
        video_weight=0.7,
        thumbnail_weight=0.3,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        vid_path = tmp_path / "benchmark_vid.mp4"
        thumb_path = tmp_path / "thumbnail.jpg"

        # Generate realistic local MP4 video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (640, 360))
        for i in range(24):
            frame = np.full((360, 640, 3), (i * 10) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()

        # Generate thumbnail file
        img = Image.new("RGB", (640, 360), color=(100, 150, 200))
        img.save(thumb_path)

        # Warmup
        _ = service.embed_video(video_path=vid_path, thumbnail_path=thumb_path)

        t_sample_list, t_pre_list, t_infer_list, t_agg_list = [], [], [], []
        t_thumb_list, t_vid_tot_list, t_fuse_list = [], [], []

        for _ in range(num_runs):
            # 1. Frame Sampling
            t0 = time.perf_counter()
            s_res = sampler.sample_frames(vid_path, num_frames=8)
            t_sample = (time.perf_counter() - t0) * 1000
            t_sample_list.append(t_sample)

            # 2. Image Preprocessing
            t0 = time.perf_counter()
            frames_t = processor.preprocess_frames(s_res.frames)
            t_pre = (time.perf_counter() - t0) * 1000
            t_pre_list.append(t_pre)

            # 3. ResNet frame inference (8 frames through backbone + projection)
            frames_batch = frames_t.unsqueeze(0).to(device)
            B, T, C, H, W = frames_batch.shape
            flat = frames_batch.view(B * T, C, H, W)
            t0 = time.perf_counter()
            with torch.no_grad():
                feat_2048 = encoder.extract_backbone_features(flat)
                feat_512 = encoder.projection(feat_2048)
                feat_512 = feat_512.view(B, T, 512)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_infer = (time.perf_counter() - t0) * 1000
            t_infer_list.append(t_infer)

            # 4. Frame aggregation
            t0 = time.perf_counter()
            with torch.no_grad():
                video_emb = encoder.aggregate_frames(feat_512, method="mean")
                video_emb = torch.nn.functional.normalize(video_emb, p=2, dim=-1)
            t_agg = (time.perf_counter() - t0) * 1000
            t_agg_list.append(t_agg)

            # 5. Thumbnail inference
            t0 = time.perf_counter()
            thumb_res = thumb_encoder.encode_thumbnail(thumb_path)
            thumb_emb = torch.nn.functional.normalize(thumb_res.embedding.to(device), p=2, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_thumb = (time.perf_counter() - t0) * 1000
            t_thumb_list.append(t_thumb)

            # 6. Video visual embedding total latency
            t0 = time.perf_counter()
            _ = service.embed_video(video_path=vid_path, thumbnail_path=None, auto_discover_thumbnail=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_vid_tot = (time.perf_counter() - t0) * 1000
            t_vid_tot_list.append(t_vid_tot)

            # 7. Video + thumbnail visual fusion latency
            t0 = time.perf_counter()
            with torch.no_grad():
                fused = 0.7 * video_emb + 0.3 * thumb_emb
                fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
                _ = torch.nan_to_num(fused, nan=0.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fuse = (time.perf_counter() - t0) * 1000
            t_fuse_list.append(t_fuse)

        avg_sample = sum(t_sample_list) / num_runs
        avg_pre = sum(t_pre_list) / num_runs
        avg_infer = sum(t_infer_list) / num_runs
        avg_agg = sum(t_agg_list) / num_runs
        avg_thumb = sum(t_thumb_list) / num_runs
        avg_vid_tot = sum(t_vid_tot_list) / num_runs
        avg_fuse = sum(t_fuse_list) / num_runs

        print("\n" + "=" * 60)
        print("  PHASE 6 — VISUAL & THUMBNAIL PIPELINE LATENCY BREAKDOWN")
        print(f"  Device: {device} | Iterations: {num_runs}")
        print("=" * 60)
        print(f"  1. Frame Sampling (OpenCV, 8 frames from MP4): {avg_sample:8.2f} ms")
        print(f"  2. Image Preprocessing (ImageNet 224x224):     {avg_pre:8.2f} ms")
        print(f"  3. ResNet-50 Frame Inference (2048->512):      {avg_infer:8.2f} ms")
        print(f"  4. Frame Aggregation (Mean Pooling):           {avg_agg:8.2f} ms")
        print(f"  5. Thumbnail Inference (ImageNet + ResNet-50): {avg_thumb:8.2f} ms")
        print(f"  6. Total Video Visual Embedding Latency:       {avg_vid_tot:8.2f} ms")
        print(f"  7. Video + Thumbnail Visual Fusion Latency:    {avg_fuse:8.2f} ms")
        print("  --------------------------------------------------")
        print(f"  Total Video + Thumbnail Latency:               {avg_vid_tot + avg_thumb + avg_fuse:8.2f} ms")
        print("=" * 60 + "\n")

        return {
            "visual_frame_sampling_ms": avg_sample,
            "visual_image_preprocessing_ms": avg_pre,
            "visual_resnet_inference_ms": avg_infer,
            "visual_frame_aggregation_ms": avg_agg,
            "visual_thumbnail_inference_ms": avg_thumb,
            "visual_video_total_ms": avg_vid_tot,
            "visual_fusion_ms": avg_fuse,
        }


def benchmark_audio_pipeline():
    """
    Measures Phase 7 Audio Intelligence Pipeline latency:
    1. Audio loading and validation
    2. Resampling (44.1 kHz -> 16 kHz)
    3. Audio chunking (windowing)
    4. Wav2Vec2 forward inference (backbone)
    5. Sequence pooling (temporal aggregation)
    6. 512-D projection & L2 normalization
    7. Audio embedding total latency
    8. Optional speech-recognition latency (separate)
    """
    from src.features.audio_processor import AudioProcessor
    from src.features.audio_chunker import AudioChunker
    from src.features.audio_encoder import AudioEncoder
    from src.features.speech_recognizer import OptionalSpeechRecognizer
    from src.services.multimodal_inference import AudioEmbeddingService
    import tempfile
    import scipy.io.wavfile as wavfile
    import numpy as np

    settings = get_settings()
    device = torch.device("cuda" if (torch.cuda.is_available() and settings.DEVICE in ("auto", "cuda")) else "cpu")

    processor = AudioProcessor(target_sample_rate=16000)
    chunker = AudioChunker(chunk_duration_seconds=20.0, overlap_seconds=2.0, sample_rate=16000)
    encoder = AudioEncoder(feature_dim=512, pretrained=False).to(device)
    encoder.eval()
    audio_service = AudioEmbeddingService(audio_encoder=encoder, device=device, feature_dim=512)
    speech_recognizer = OptionalSpeechRecognizer(enabled=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create 44.1 kHz 5-second WAV file
        sr_44k = 44100
        t = np.linspace(0, 5.0, int(sr_44k * 5.0), endpoint=False, dtype=np.float32)
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
        test_wav = tmp_path / "bench_audio_44k.wav"
        wavfile.write(str(test_wav), sr_44k, sine)

        # Warmup
        _ = audio_service.encode(test_wav)

        num_runs = 10
        t_load_list, t_resample_list, t_chunk_list = [], [], []
        t_forward_list, t_pool_list, t_proj_list, t_total_list, t_speech_list = [], [], [], [], []

        raw_data, orig_sr, orig_ch = processor._load_audio_file(test_wav)
        waveform_16k_np = processor.resample(raw_data.astype(np.float32) / 32768.0, orig_sr, 16000)
        waveform_16k = torch.from_numpy(waveform_16k_np).float()

        chunk_res = chunker.chunk_waveform(waveform_16k)
        chunks = chunk_res.chunks.to(device)

        for _ in range(num_runs):
            # 1. Loading & validation
            t0 = time.perf_counter()
            _ = processor._load_audio_file(test_wav)
            t_load_list.append((time.perf_counter() - t0) * 1000)

            # 2. Resampling
            t0 = time.perf_counter()
            _ = processor.resample(raw_data.astype(np.float32) / 32768.0, orig_sr, 16000)
            t_resample_list.append((time.perf_counter() - t0) * 1000)

            # 3. Audio chunking
            t0 = time.perf_counter()
            _ = chunker.chunk_waveform(waveform_16k)
            t_chunk_list.append((time.perf_counter() - t0) * 1000)

            # 4. Wav2Vec2 forward inference
            t0 = time.perf_counter()
            with torch.no_grad():
                out = encoder.backbone(chunks)
                hidden_states = out.last_hidden_state
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward_list.append((time.perf_counter() - t0) * 1000)

            # 5. Sequence pooling
            t0 = time.perf_counter()
            with torch.no_grad():
                pooled = hidden_states.mean(dim=1)
            t_pool_list.append((time.perf_counter() - t0) * 1000)

            # 6. 512-D projection & L2 normalization
            t0 = time.perf_counter()
            with torch.no_grad():
                proj = encoder.projection(pooled)
                _ = torch.nn.functional.normalize(proj, p=2, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_proj_list.append((time.perf_counter() - t0) * 1000)

            # 7. Audio embedding total latency
            t0 = time.perf_counter()
            _ = audio_service.encode(test_wav)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_total_list.append((time.perf_counter() - t0) * 1000)

            # 8. Optional speech recognition latency (separate)
            t0 = time.perf_counter()
            _ = speech_recognizer.transcribe(waveform_16k)
            t_speech_list.append((time.perf_counter() - t0) * 1000)

    avg_load = sum(t_load_list) / num_runs
    avg_resample = sum(t_resample_list) / num_runs
    avg_chunk = sum(t_chunk_list) / num_runs
    avg_forward = sum(t_forward_list) / num_runs
    avg_pool = sum(t_pool_list) / num_runs
    avg_proj = sum(t_proj_list) / num_runs
    avg_total = sum(t_total_list) / num_runs
    avg_speech = sum(t_speech_list) / num_runs

    print("\n" + "=" * 60)
    print("  PHASE 7 — AUDIO & SPEECH PIPELINE LATENCY BREAKDOWN")
    print(f"  Device: {device} | Iterations: {num_runs}")
    print("=" * 60)
    print(f"  1. Audio Loading & Validation (WAV file):       {avg_load:8.2f} ms")
    print(f"  2. Resampling (44.1 kHz -> 16 kHz polyphase):   {avg_resample:8.2f} ms")
    print(f"  3. Audio Chunking (Windowing + Masks):          {avg_chunk:8.2f} ms")
    print(f"  4. Wav2Vec2 Forward Inference:                  {avg_forward:8.2f} ms")
    print(f"  5. Sequence Pooling (Mean Pooling):             {avg_pool:8.2f} ms")
    print(f"  6. 512-D Projection & L2 Normalization:         {avg_proj:8.2f} ms")
    print("  --------------------------------------------------")
    print(f"  7. Total Audio Embedding Latency:               {avg_total:8.2f} ms")
    print(f"  8. Optional Speech-Recognition (Disabled):      {avg_speech:8.4f} ms")
    print("=" * 60 + "\n")

    return {
        "audio_loading_ms": avg_load,
        "audio_resampling_ms": avg_resample,
        "audio_chunking_ms": avg_chunk,
        "wav2vec2_forward_ms": avg_forward,
        "sequence_pooling_ms": avg_pool,
        "projection_and_norm_ms": avg_proj,
        "audio_embedding_total_ms": avg_total,
        "speech_recognition_ms": avg_speech,
    }


if __name__ == "__main__":
    benchmark()
    benchmark_visual_pipeline()
    benchmark_audio_pipeline()

