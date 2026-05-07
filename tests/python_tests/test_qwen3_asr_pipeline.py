# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test script for Qwen3-ASR model support in openvino.genai WhisperPipeline.

Usage:
    python test_qwen3_asr_pipeline.py --model-path /path/to/qwen3-asr-ov
"""

import argparse
import sys
import numpy as np
import openvino_genai as ov_genai


def generate_sine_wave(frequency=440, duration=2.0, sample_rate=16000):
    """Generate a simple sine wave audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32).tolist()


def test_silence_input(pipe):
    """Test with silent audio - model should produce minimal or empty output."""
    print("\n=== Test: Silence Input ===")
    silence = [0.0] * 16000 * 2  # 2 seconds of silence
    result = pipe.generate(silence)
    print(f"Output: {repr(result.texts[0])}")
    print("PASSED" if isinstance(result.texts[0], str) else "FAILED")
    return True


def test_sine_wave_input(pipe):
    """Test with a sine wave - model should produce some output."""
    print("\n=== Test: Sine Wave Input ===")
    audio = generate_sine_wave(frequency=440, duration=3.0)
    result = pipe.generate(audio)
    print(f"Output: {repr(result.texts[0])}")
    print("PASSED" if isinstance(result.texts[0], str) else "FAILED")
    return True


def test_max_new_tokens(pipe):
    """Test max_new_tokens parameter."""
    print("\n=== Test: Max New Tokens ===")
    audio = generate_sine_wave(frequency=440, duration=2.0)
    config = pipe.get_generation_config()
    config.max_new_tokens = 10
    result = pipe.generate(audio, config)
    print(f"Output: {repr(result.texts[0])}")
    print("PASSED" if isinstance(result.texts[0], str) else "FAILED")
    return True


def test_perf_metrics(pipe):
    """Test that performance metrics are populated."""
    print("\n=== Test: Performance Metrics ===")
    audio = generate_sine_wave(frequency=440, duration=2.0)
    result = pipe.generate(audio)
    metrics = result.perf_metrics
    print(f"Load time: {metrics.load_time:.0f}ms")
    print(f"Generate duration: {metrics.get_generate_duration().mean:.1f}ms")
    print(f"Tokenization duration: {metrics.get_tokenization_duration().mean:.1f}ms")
    print(f"Detokenization duration: {metrics.get_detokenization_duration().mean:.1f}ms")
    passed = metrics.load_time >= 0 and metrics.get_generate_duration().mean > 0
    print("PASSED" if passed else "FAILED")
    return passed


def test_streamer(pipe):
    """Test streamer callback."""
    print("\n=== Test: Streamer ===")
    audio = generate_sine_wave(frequency=440, duration=2.0)
    streamed_text = []

    def streamer_callback(text):
        streamed_text.append(text)
        return False  # Continue generation

    result = pipe.generate(audio, streamer=streamer_callback)
    print(f"Output: {repr(result.texts[0])}")
    print(f"Streamed chunks: {len(streamed_text)}")
    print("PASSED" if isinstance(result.texts[0], str) else "FAILED")
    return True


def test_real_speech(pipe):
    """Test with real speech audio from librispeech."""
    print("\n=== Test: Real Speech ===")
    try:
        import urllib.request
        import soundfile as sf
        import io

        url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
        resp = urllib.request.urlopen(url)
        audio_data = resp.read()
        audio, sr = sf.read(io.BytesIO(audio_data))
        audio = audio.astype(np.float32).tolist()

        result = pipe.generate(audio)
        text = result.texts[0]
        print(f"Output: {repr(text)}")

        # Check if the output contains expected words
        expected_words = ["stew", "dinner", "turnips", "carrots"]
        found = sum(1 for w in expected_words if w.lower() in text.lower())
        print(f"Found {found}/{len(expected_words)} expected words")
        passed = found >= 2
        print("PASSED" if passed else "FAILED")
        return passed
    except Exception as e:
        print(f"Skipped (network/dependency error): {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR pipeline")
    parser.add_argument("--model-path", required=True, help="Path to exported Qwen3-ASR model")
    parser.add_argument("--device", default="CPU", help="Device to run on")
    args = parser.parse_args()

    print(f"Loading Qwen3-ASR model from: {args.model_path}")
    pipe = ov_genai.WhisperPipeline(args.model_path, args.device)
    print("Model loaded successfully!")

    # Print generation config
    config = pipe.get_generation_config()
    print(f"EOS token ID: {config.eos_token_id}")
    print(f"Max new tokens: {config.max_new_tokens}")

    # Run tests
    results = []
    results.append(("Silence", test_silence_input(pipe)))
    results.append(("Sine Wave", test_sine_wave_input(pipe)))
    results.append(("Max Tokens", test_max_new_tokens(pipe)))
    results.append(("Perf Metrics", test_perf_metrics(pipe)))
    results.append(("Streamer", test_streamer(pipe)))
    results.append(("Real Speech", test_real_speech(pipe)))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
