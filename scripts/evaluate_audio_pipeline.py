"""
Offline Evaluation Script for Phase 7 Audio & Speech Intelligence Pipeline.

Performs deterministic pipeline sanity verification:
- Output representation dimensionality: (1, 512)
- L2 unit sphere normalization contract
- Deterministic repeatability on identical waveform
- Non-finite (NaN / Inf) defense and safety
- Silence detection and zero-vector return
- Short recording (< 5s) vs. Long recording (> 20s) multi-chunk aggregation
- Acoustic feature calculations (RMS energy, silence ratio)
- Missing-modality recovery under multimodal fusion

Note: In accordance with Project Instructions, this script performs empirical
pipeline validation and sanity checks. It does NOT fabricate ground-truth semantic
recommendation relevance scores without a verified human-labelled audio dataset.
"""

from pathlib import Path
import sys
import tempfile
import time

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from src.features.audio_encoder import AudioEncoder
from src.features.audio_processor import AudioProcessor
from src.features.audio_chunker import AudioChunker
from src.features.speech_recognizer import OptionalSpeechRecognizer
from src.services.multimodal_inference import AudioEmbeddingService, AudioEmbeddingResult


def evaluate_audio_pipeline():
    print("=" * 70)
    print("  PHASE 7 — AUDIO & SPEECH INTELLIGENCE OFFLINE EVALUATION")
    print("  Mode: Pipeline Sanity Check & Representation Robustness")
    print("=" * 70)

    device = torch.device("cpu")
    encoder = AudioEncoder(feature_dim=512, pretrained=False)
    encoder.eval()
    service = AudioEmbeddingService(audio_encoder=encoder, device=device, feature_dim=512)
    processor = AudioProcessor(target_sample_rate=16000)

    results = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Output Dimensions & L2 Normalization
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
        test_wav = tmp_path / "eval_2s.wav"
        wavfile.write(str(test_wav), sr, sine)

        res_2s = service.encode(test_wav)
        dim_ok = res_2s.embedding.shape == (1, 512)
        norm = torch.norm(res_2s.embedding, p=2, dim=-1).item()
        l2_ok = abs(norm - 1.0) < 1e-4
        print(f"[*] Dimension Check (Expected: (1, 512)):          {res_2s.embedding.shape} -> {'PASS' if dim_ok else 'FAIL'}")
        print(f"[*] L2 Unit Sphere Normalization (Norm == 1.0):   {norm:.6f} -> {'PASS' if l2_ok else 'FAIL'}")
        results["dimensions_pass"] = dim_ok
        results["l2_norm_pass"] = l2_ok

        # 2. Deterministic Repeatability
        res_repeat = service.encode(test_wav)
        diff = torch.max(torch.abs(res_2s.embedding - res_repeat.embedding)).item()
        repeat_ok = diff < 1e-6
        print(f"[*] Determinism & Repeatability (Max Diff == 0.0): {diff:.2e} -> {'PASS' if repeat_ok else 'FAIL'}")
        results["repeatability_pass"] = repeat_ok

        # 3. Non-Finite (NaN / Inf) Defense
        bad_waveform = torch.tensor([0.5, float("nan"), -0.5, float("inf"), 0.2] + [0.0] * 15995)
        res_bad = service.encode(bad_waveform)
        has_nan = torch.isnan(res_bad.embedding).any().item()
        has_inf = torch.isinf(res_bad.embedding).any().item()
        nan_ok = (not has_nan) and (not has_inf)
        print(f"[*] NaN / Inf Shielding:                           No NaN/Inf in output -> {'PASS' if nan_ok else 'FAIL'}")
        results["nan_inf_pass"] = nan_ok

        # 4. Silence Detection & Handling
        silent_wav = tmp_path / "eval_silent.wav"
        wavfile.write(str(silent_wav), sr, np.zeros(16000, dtype=np.int16))
        res_silence = service.encode(silent_wav)
        silence_ok = (not res_silence.is_available) and (res_silence.embedding == 0.0).all().item()
        print(f"[*] Silence Detection & Zero-Vector Return:        is_available={res_silence.is_available} -> {'PASS' if silence_ok else 'FAIL'}")
        results["silence_handling_pass"] = silence_ok

        # 5. Short (< 5s) vs Long (> 20s) Recording Chunking
        t_long = np.linspace(0, 25.0, sr * 25, endpoint=False, dtype=np.float32)
        sine_long = (0.5 * np.sin(2 * np.pi * 440.0 * t_long) * 32767).astype(np.int16)
        long_wav = tmp_path / "eval_long_25s.wav"
        wavfile.write(str(long_wav), sr, sine_long)

        res_long = service.encode(long_wav)
        long_ok = res_long.is_available and res_long.num_chunks > 1 and res_long.embedding.shape == (1, 512)
        print(f"[*] Long Audio Chunking (>20s windowing):          {res_long.num_chunks} chunks -> {'PASS' if long_ok else 'FAIL'}")
        results["long_chunking_pass"] = long_ok

        # 6. Acoustic Signal Feature Intelligence
        proc_res = processor.process_file(test_wav)
        rms = proc_res.rms_energy
        silence_ratio = proc_res.silence_ratio
        acoustic_ok = rms > 0.0 and silence_ratio < 0.1
        print(f"[*] Acoustic Signal Diagnostics (RMS, Silence):    RMS={rms:.4f}, SilRatio={silence_ratio:.2f} -> {'PASS' if acoustic_ok else 'FAIL'}")
        results["acoustic_features_pass"] = acoustic_ok

        # 7. Missing Modality Handling
        missing_res = service.encode(None)
        missing_ok = (not missing_res.is_available) and (missing_res.embedding == 0.0).all().item()
        print(f"[*] Missing Modality Handling (None input):        is_available={missing_res.is_available} -> {'PASS' if missing_ok else 'FAIL'}")
        results["missing_modality_pass"] = missing_ok

        # 8. Optional Speech Recognizer Default
        asr = OptionalSpeechRecognizer(enabled=False)
        asr_res = asr.transcribe(test_wav)
        asr_ok = (not asr_res.is_available) and (asr_res.text is None)
        print(f"[*] Optional Speech Recognizer (Default Disabled): is_available={asr_res.is_available} -> {'PASS' if asr_ok else 'FAIL'}")
        results["speech_recognizer_default_pass"] = asr_ok

    all_passed = all(results.values())
    print("-" * 70)
    print(f"  OVERALL EVALUATION STATUS: {'PASS' if all_passed else 'FAIL'} ({sum(results.values())}/{len(results)} checks passed)")
    print("=" * 70)
    return results


if __name__ == "__main__":
    evaluate_audio_pipeline()
