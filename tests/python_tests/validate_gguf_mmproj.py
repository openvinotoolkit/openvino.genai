# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GGUF multimodal acceptance against a pinned llama.cpp CPU oracle.

Build gguf_mmproj_oracle.cpp against REFERENCE_REVISION with LLAMA_REFERENCE_REVISION set
to that revision. No llama.cpp production dependency.

Use Q4_0 language checkpoints for quantized accuracy acceptance. Q4_K_M conversion
has an expected accuracy loss relative to llama.cpp pending a plugin-side fix;
runs with those checkpoints remain diagnostic and retain the same thresholds.

Use --require-q4-0 to verify tensor types; checkpoint filenames alone do not specify
every tensor's precision. The pinned gguf_reference genai-gguf-quantize tool can
expand a checkpoint to F16, then quantize that expansion to a pure Q4_0 fixture.
Record both files' hashes in --reference-manifest. Requantization creates a new
fixture; it does not establish parity for the original mixed-weight checkpoint.
"""
import argparse
import json
import hashlib
import os
import traceback
import subprocess  # nosec B404
import tempfile
import wave
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
from PIL import Image

REFERENCE_REVISION = "03fa73cb27f5c251b9528489b18d303b1366aca4"


class Tokens(genai.StreamerBase):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def write(self, tokens):
        self.tokens.extend(tokens if isinstance(tokens, list) else [tokens])
        return genai.StreamingStatus.RUNNING

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", type=Path)
    parser.add_argument("mmproj", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--reference-revision", default=REFERENCE_REVISION,
                        help="Exact expected oracle revision, including a pinned fork for additional GGUF formats")
    parser.add_argument("--reference-language", type=Path, help="Alternative language checkpoint for reference inference")
    parser.add_argument("--reference-mmproj", type=Path, help="Alternative projector checkpoint for reference inference")
    parser.add_argument("--reference-kind", choices=("quantized", "represented-f32", "publisher-high-precision"),
                        default="quantized")
    parser.add_argument("--reference-manifest", type=Path, help="Pinned source and precision metadata for the reference")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, help="Source manifest for an immutable runtime snapshot")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video-frames", type=int, default=0, help="Also compare a short synthetic video")
    parser.add_argument("--audio", type=Path, help="Also compare audio and mixed image/audio requests (16 kHz mono WAV)")
    parser.add_argument("--api-checks", action="store_true", help="Check beam search, modern chat, streaming cancellation and reset")
    parser.add_argument("--multi-media", action="store_true", help="Compare two images and, when available, two audio inputs")
    parser.add_argument("--audio-boundaries", action="store_true", help="Compare audio immediately before and after the 30-second chunk boundary")
    parser.add_argument("--chat", action="store_true", help="Also compare a cached image chat follow-up")
    parser.add_argument("--family", choices=("gemma3", "gemma4", "qwen35", "muse", "lfm2", "lfm2_audio"), default="gemma3")
    parser.add_argument("--attention-backend", choices=("SDPA", "PA"), default="SDPA")
    parser.add_argument("--kv-cache-precision", choices=("f16", "f32"), default="f16")
    parser.add_argument("--full-coverage", action="store_true",
                        help="Require all family modalities, chat and API acceptance cases")
    parser.add_argument("--require-q4-0", action="store_true",
                        help="Reject language checkpoints containing quantized tensor types other than Q4_0")
    args = parser.parse_args()
    language_tensor_types = None
    if args.require_q4_0:
        from collections import Counter
        from gguf import GGUFReader
        language_tensor_types = dict(Counter(t.tensor_type.name for t in GGUFReader(str(args.language)).tensors))
        if "Q4_0" not in language_tensor_types or set(language_tensor_types) - {"F32", "F16", "Q4_0"}:
            parser.error(f"Q4_0 accuracy acceptance requires F32/F16/Q4_0 tensors, got {language_tensor_types}")
    revision = subprocess.check_output([str(args.oracle.resolve()), "--revision"], text=True).strip()
    if revision != args.reference_revision:
        parser.error(f"Oracle revision {revision!r} does not match {args.reference_revision}")
    if args.video_frames < 0:
        parser.error("--video-frames must be nonnegative")
    if args.video_frames and args.family not in ("gemma4", "qwen35", "muse"):
        parser.error("Video oracle assembly is currently supported for Gemma4, Qwen3.5 and Muse Glimmer")
    if args.full_coverage:
        if not (args.multi_media and args.chat and args.api_checks):
            parser.error("--full-coverage requires --multi-media, --chat and --api-checks")
        if args.family not in ("gemma3", "lfm2", "lfm2_audio") and not args.video_frames:
            parser.error("--full-coverage requires --video-frames for this family")
        if args.family == "gemma4" and not (args.audio and args.audio_boundaries):
            parser.error("--full-coverage requires --audio and --audio-boundaries for Gemma4")
    report = {"language": str(args.language), "mmproj": str(args.mmproj),
              "reference_revision": args.reference_revision, "family": args.family,
              "reference_language": str(args.reference_language or args.language),
              "reference_mmproj": str(args.reference_mmproj or args.mmproj),
              "reference_kind": args.reference_kind,
              "q4_k_zp_f16": os.environ.get("OV_GGUF_Q4_K_ZP_F16"),
              "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
              "video_frames": args.video_frames, "image": str(args.image) if args.image else None,
              "audio": str(args.audio) if args.audio else None,
              "multi_media": args.multi_media, "audio_boundaries": args.audio_boundaries,
              "attention_backend": args.attention_backend,
              "kv_cache_precision": args.kv_cache_precision,
              "language_tensor_types": language_tensor_types,
              "openvino_version": ov.get_version(), "genai_version": genai.__version__,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "full_coverage": args.full_coverage,
              "cases": [], "chat_checked": args.chat, "passed": False, "completed": False}
    if args.runtime_manifest:
        report["runtime_sources"] = json.loads(args.runtime_manifest.read_text())
    if args.reference_manifest:
        report["reference_sources"] = json.loads(args.reference_manifest.read_text())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        (run_lfm2_audio if args.family == "lfm2_audio" else run)(args, report)
        validate_coverage(args, report)
        report["completed"] = True
        report["passed"] = (report["request_reset_matches"] and report["chat_reset_matches"] and
                            all(c["passed"] for c in report.get("api_checks", {}).values()) and
                            all(case_passed(c) for c in report["cases"]))
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        save_report(args, report)
    return 0 if report["passed"] else 1


def validate_coverage(args, report):
    if getattr(args, "family", None) == "lfm2_audio":
        expected = {"text", "audio", "multi_audio", "reversed_audio", "audio_chat", "audio_reset"}
        modalities = [case["modality"] for case in report["cases"]]
        if len(modalities) != len(set(modalities)) or set(modalities) != expected:
            raise ValueError(f"Expected {expected}, got {modalities}")
        api = {"beam_search", "cancel_and_reset", "invalid_audio"} if args.api_checks else set()
        if set(report.get("api_checks", {})) != api:
            raise ValueError(f"Expected API checks {api}")
        report["required_modalities"] = sorted(expected)
        report["required_api_checks"] = sorted(api)
        return
    expected = {"text", "image"}
    api = set()
    if args.video_frames:
        expected.add("video")
    if args.multi_media:
        expected.add("multi_image")
        if getattr(args, "family", None) == "lfm2":
            expected.add("reversed_image")
    if args.chat:
        expected.add("image_chat")
    if args.audio:
        expected.update(("audio", "mixed"))
        if args.multi_media:
            expected.add("multi_audio")
        if args.audio_boundaries:
            expected.update(("audio_samples_479999", "audio_samples_480321"))
        if args.chat:
            expected.add("audio_chat")
    if args.api_checks:
        api.update(("beam_search", "cancel_and_reset"))
        if args.chat:
            expected.update(("modern_image", "modern_image_chat"))
            api.add("modern_image_chat")
            if args.audio:
                expected.update(("modern_audio", "modern_audio_chat"))
                api.add("modern_audio_chat")
    modalities = [case["modality"] for case in report["cases"]]
    if len(modalities) != len(set(modalities)) or set(modalities) != expected:
        raise ValueError(f"Expected modalities {sorted(expected)}, got {modalities}")
    if set(report.get("api_checks", {})) != api:
        raise ValueError(f"Expected API checks {sorted(api)}, got {list(report.get('api_checks', {}))}")
    report["required_modalities"] = sorted(expected)
    report["required_api_checks"] = sorted(api)


def case_passed(case):
    tokens, choices = case["tokens"], case["reference_choices_on_same_history"]
    if not tokens or len(tokens) != len(choices):
        return False
    return tokens[0] == choices[0] and sum(a == b for a, b in zip(tokens, choices)) / len(tokens) >= .9


def save_report(args, report):
    args.report.write_text(json.dumps(report, indent=2) + "\n")


def run_lfm2_audio(args, report):
    if not args.audio:
        raise ValueError("LFM2 audio acceptance requires --audio")
    pipe = genai.VLMPipeline(str(args.language), "CPU", mmproj_path=str(args.mmproj),
                            ATTENTION_BACKEND=args.attention_backend,
                            INFERENCE_PRECISION_HINT="f32", KV_CACHE_PRECISION=args.kv_cache_precision,
                            DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_NUM_THREADS=4)
    tokenizer = pipe.get_tokenizer()
    with wave.open(str(args.audio)) as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1 and source.getsampwidth() == 2
        waveform = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    def generate(prompt, **kwargs):
        stream = Tokens()
        result = pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=stream, **kwargs)
        return stream.tokens, result.texts[0]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        def compare(name, messages, result, samples):
            tokens, text = result
            files = []
            for i, samples_i in enumerate(samples):
                path = directory / f"audio-{i}.wav"
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes((samples_i * 32768).astype("<i2").tobytes())
                files.append(str(path))
            prompt = directory / "prompt.txt"
            prompt.write_text(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            history = directory / "history.txt"
            history.write_text(" ".join(map(str, tokens)))
            command = [str(args.oracle.resolve()), str((args.reference_language or args.language).resolve()),
                       str((args.reference_mmproj or args.mmproj).resolve()), ";".join(files) or "-",
                       str(prompt), str(history)]
            if args.kv_cache_precision == "f32":
                command.append("--kv-cache-f32")
            process = subprocess.run(command, capture_output=True, text=True, errors="replace")
            (args.report.parent / f"{args.report.stem}-{name}.log").write_text(process.stderr)
            process.check_returncode()
            reference = list(map(int, next(line for line in process.stdout.splitlines() if line.startswith("CHOICES")).split()[1:]))
            case = {"modality": name, "text": text, "tokens": tokens,
                    "reference_choices_on_same_history": reference,
                    "first_token_matches": bool(tokens) and reference[0] == tokens[0],
                    "matching_choice_fraction": sum(a == b for a, b in zip(tokens, reference)) / max(len(tokens), 1)}
            report["cases"].append(case)
            save_report(args, report)
            print(json.dumps(case), flush=True)
        base = "<audio>\nTranscribe the audio."
        reference_base = "<__media__>\nTranscribe the audio."
        first = generate(base, audios=[ov.Tensor(waveform)])
        text = generate("What is 2 plus 2?")
        compare("text", [{"role": "user", "content": "What is 2 plus 2?"}], text, [])
        compare("audio", [{"role": "user", "content": reference_base}], first, [waveform])
        second = np.ascontiguousarray(waveform[::-1])
        for reverse in (False, True):
            samples = [second, waveform] if reverse else [waveform, second]
            prompt = "<ov_genai_audio_1><ov_genai_audio_0> Compare the clips." if reverse else "<audio><audio> Compare the clips."
            supplied = [waveform, second]
            compare("reversed_audio" if reverse else "multi_audio",
                    [{"role": "user", "content": "<__media__><__media__> Compare the clips."}],
                    generate(prompt, audios=[ov.Tensor(x) for x in supplied]), samples)
        chat = genai.ChatHistory([{"role": "user", "content": base}])
        _, first_text = generate(chat, audios=[ov.Tensor(waveform)])
        chat.append({"role": "assistant", "content": first_text})
        chat.append({"role": "user", "content": "What did the speaker say?"})
        compare("audio_chat", [{"role": "user", "content": reference_base},
                               {"role": "assistant", "content": first_text},
                               {"role": "user", "content": "What did the speaker say?"}], generate(chat), [waveform])
        repeated = generate(base, audios=[ov.Tensor(waveform)])
        compare("audio_reset", [{"role": "user", "content": reference_base}], repeated, [waveform])
        report["request_reset_matches"] = repeated[0] == first[0]
        report["chat_reset_matches"] = True
        if args.api_checks:
            checks = report["api_checks"] = {}
            def check(name, function):
                try:
                    function()
                    checks[name] = {"passed": True}
                except Exception:
                    checks[name] = {"passed": False, "error": traceback.format_exc()}
                save_report(args, report)
            def beams():
                result = pipe.generate(base, audios=[ov.Tensor(waveform)], num_beams=3,
                                       num_return_sequences=2, max_new_tokens=8, min_new_tokens=8, do_sample=False)
                assert len(result.texts) == 2 and np.isfinite(result.scores).all()
            def cancel():
                class Cancel(Tokens):
                    def write(self, tokens):
                        super().write(tokens)
                        return genai.StreamingStatus.CANCEL
                pipe.generate(base, audios=[ov.Tensor(waveform)], max_new_tokens=20, streamer=Cancel())
                assert generate("What is 2 plus 2?")[0] == text[0]
            def invalid():
                for value in (np.zeros(159, np.float32), np.zeros((2, 160), np.float32)):
                    try:
                        pipe.generate(base, audios=[ov.Tensor(value)], max_new_tokens=1)
                    except RuntimeError:
                        continue
                    raise AssertionError("Invalid audio was accepted")
                assert generate("What is 2 plus 2?")[0] == text[0]
            check("beam_search", beams)
            check("cancel_and_reset", cancel)
            check("invalid_audio", invalid)


def run(args, report):
    image_marker = {"gemma3": "<start_of_image>", "gemma4": "<|image|>", "muse": "<|image|>", "lfm2": "<image>",
                    "qwen35": "<|vision_start|><|image_pad|><|vision_end|>"}[args.family]
    image_prompt = "<ov_genai_image_0>\nDescribe the image."
    pipe = genai.VLMPipeline(str(args.language), "CPU", mmproj_path=str(args.mmproj),
                            ATTENTION_BACKEND=args.attention_backend,
                            INFERENCE_PRECISION_HINT="f32", KV_CACHE_PRECISION=args.kv_cache_precision,
                            DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
                            INFERENCE_NUM_THREADS=4)
    tokenizer = pipe.get_tokenizer()
    if args.image:
        pixels = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        pixels = np.zeros((64, 64, 3), np.uint8)
        pixels[..., 0] = 255
    cases = report["cases"]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        image_file = directory / "image.png"
        Image.fromarray(pixels).save(image_file)

        image = [str(image_file)]
        audio = [str(args.audio.resolve())] if args.audio else []

        def generate(prompt, **kwargs):
            """Greedy 20-token generation; returns (tokens, text)."""
            stream = Tokens()
            result = pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=stream, **kwargs)
            return stream.tokens, result.texts[0]

        def compare(messages, tokens, text, modality, media=(), merge_frames=False, muse_video=False):
            """Replay `tokens` through the oracle on the same history and score the agreement."""
            # GenAI expands image markers into embeddings. mtmd uses its own marker and adds
            # the same Gemma3 begin/end-image tokens around the reference encoder output.
            (directory / "prompt.txt").write_text(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            (directory / "history.txt").write_text(" ".join(map(str, tokens)))
            oracle_options = (["--merge-frames"] if merge_frames else []) + (["--muse-video"] if muse_video else [])
            if args.kv_cache_precision == "f32":
                oracle_options.append("--kv-cache-f32")
            process = subprocess.run([str(args.oracle.resolve()), str((args.reference_language or args.language).resolve()),
                str((args.reference_mmproj or args.mmproj).resolve()), ";".join(media) if media else "-",
                str(directory / "prompt.txt"), str(directory / "history.txt")] + oracle_options, capture_output=True, text=True, errors="replace")
            (args.report.parent / f"{args.report.stem}-{modality}.log").write_text(process.stderr)
            process.check_returncode()
            choices = next(line for line in process.stdout.splitlines() if line.startswith("CHOICES"))
            reference = list(map(int, choices.split()[1:]))
            assert len(reference) == len(tokens) and reference
            case = {"modality": modality, "text": text, "tokens": tokens,
                    "reference_choices_on_same_history": reference,
                    "first_token_matches": reference[0] == tokens[0],
                    "matching_choice_fraction": sum(a == b for a, b in zip(reference, tokens)) / len(reference)}
            cases.append(case)
            save_report(args, report)
            print(json.dumps(case), flush=True)
            return case

        for with_image in (False, True):
            prompt = image_prompt if with_image else "What is 2 plus 2?"
            kwargs = {"images": [ov.Tensor(pixels[None])]} if with_image else {}
            compare([{"role": "user", "content": prompt.replace(image_marker, "<__media__>").replace("<ov_genai_image_0>", "<__media__>").replace("<ov_genai_image_1>", "<__media__>")}],
                    *generate(prompt, **kwargs), "image" if with_image else "text", image if with_image else ())
        if args.video_frames:
            frames = np.stack([np.roll(pixels, i * 8, axis=1) for i in range(args.video_frames)])
            metadata = genai.VideoMetadata()
            metadata.fps = 2.
            files = []
            for i, frame in enumerate(frames):
                file = directory / f"frame{i}.png"
                Image.fromarray(frame).save(file)
                files.append(str(file))
            if args.family == "qwen35":
                marker = "<ov_genai_video_0>"
                reference_prompt = "".join(
                    f"<{(i + min(i + 1, len(frames) - 1)) / 4:.1f} seconds>" +
                    "<__media__>" * min(2, len(frames) - i) for i in range(0, len(frames), 2))
            elif args.family == "muse":
                marker = "<ov_genai_video_0>"
                reference_prompt = "<|vid_start|>" + "".join(
                    f"Time: {i / metadata.fps:.1f}s<__media__>" +
                    ("<|vid_frame_separator|>" if i + 1 < len(frames) else "<|vid_end|>")
                    for i in range(len(frames)))
            else:
                marker = "<ov_genai_video_0>"
                reference_prompt = " ".join(f"00:{i // 2:02d} <__media__>" for i in range(len(frames)))
            compare([{"role": "user", "content": reference_prompt + "\nDescribe the video."}],
                    *generate(marker + "\nDescribe the video.", videos=[ov.Tensor(frames)], videos_metadata=[metadata]),
                    "video", files, merge_frames=args.family == "qwen35", muse_video=args.family == "muse")
        if args.audio:
            with wave.open(str(args.audio)) as audio_file:
                assert audio_file.getframerate() == 16000 and audio_file.getnchannels() == 1
                assert audio_file.getsampwidth() == 2
                waveform = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2").astype(np.float32) / 32768
            for mixed in (False, True):
                prompt = (image_marker + "\n" if mixed else "") + "<|audio|>\nTranscribe the audio."
                kwargs = {"audios": [ov.Tensor(waveform)]}
                if mixed:
                    kwargs["images"] = [ov.Tensor(pixels[None])]
                reference_prompt = prompt.replace(image_marker, "<__media__>").replace("<ov_genai_image_0>", "<__media__>").replace("<ov_genai_image_1>", "<__media__>").replace("<|audio|>", "<__media__>")
                compare([{"role": "user", "content": reference_prompt}], *generate(prompt, **kwargs),
                        "mixed" if mixed else "audio", (image if mixed else []) + audio)
        if args.multi_media:
            second_pixels = np.ascontiguousarray(pixels.transpose(1, 0, 2))
            if args.family == "lfm2":
                second_pixels = 255 - second_pixels
            second_file = directory / "image2.png"
            Image.fromarray(second_pixels).save(second_file)
            prompt = "<ov_genai_image_0>\n<ov_genai_image_1>\nCompare these images."
            compare([{"role": "user", "content": prompt.replace(image_marker, "<__media__>").replace("<ov_genai_image_0>", "<__media__>").replace("<ov_genai_image_1>", "<__media__>")}],
                    *generate(prompt, images=[ov.Tensor(pixels[None]), ov.Tensor(second_pixels[None])]),
                    "multi_image", image + [str(second_file)])
        if args.audio and (args.multi_media or args.audio_boundaries):
            def audio_file(samples, name):
                file = directory / (name + ".wav")
                with wave.open(str(file), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes((samples * 32768).astype("<i2").tobytes())
                return str(file)

            extra_audios = []
            if args.multi_media:
                extra_audios.append(("multi_audio", [waveform, np.ascontiguousarray(waveform[::-1])]))
            if args.audio_boundaries:
                for length in (30 * 16000 - 1, 30 * 16000 + 321):
                    extra_audios.append((f"audio_samples_{length}", [np.resize(waveform, length)]))
            for name, samples in extra_audios:
                prompt = "<|audio|>\n" * len(samples) + "Transcribe the audio."
                files = [audio_file(sample, f"{name}_{i}") for i, sample in enumerate(samples)]
                compare([{"role": "user", "content": prompt.replace("<|audio|>", "<__media__>")}],
                        *generate(prompt, audios=[ov.Tensor(sample) for sample in samples]), name, files)
        if args.multi_media and args.family == "lfm2":
            compare([{"role": "user", "content": "<__media__>\n<__media__>\nCompare these images."}],
                    *generate("<ov_genai_image_1>\n<ov_genai_image_0>\nCompare these images.",
                              images=[ov.Tensor(pixels[None]), ov.Tensor(second_pixels[None])]),
                    "reversed_image", [str(second_file)] + image)
        chat_reset_matches = True
        if args.chat:
            pipe.start_chat()
            first_tokens, first_text = generate(image_prompt, images=[ov.Tensor(pixels[None])])
            chat_reset_matches = first_tokens == cases[1]["tokens"]
            report["chat_initial_tokens"] = first_tokens
            followup_prompt = "What is shown?"
            compare([{"role": "user", "content": "<__media__>\nDescribe the image."},
                     {"role": "assistant", "content": first_text},
                     {"role": "user", "content": followup_prompt}],
                    *generate(followup_prompt), "image_chat", image)
            pipe.finish_chat()
        if args.audio and args.chat:
            pipe.start_chat()
            _, first_text = generate("<|audio|>\nTranscribe the audio.", audios=[ov.Tensor(waveform)])
            followup_prompt = "What did the speaker say?"
            compare([{"role": "user", "content": "<__media__>\nTranscribe the audio."},
                     {"role": "assistant", "content": first_text},
                     {"role": "user", "content": followup_prompt}],
                    *generate(followup_prompt), "audio_chat", audio)
            pipe.finish_chat()
        if args.api_checks:
            checks = report["api_checks"] = {}

            def check(name, function):
                try:
                    function()
                    checks[name] = {"passed": True}
                except Exception:
                    checks[name] = {"passed": False, "error": traceback.format_exc()}
                finally:
                    pipe.finish_chat()
                save_report(args, report)
                print(name, checks[name], flush=True)

            audio_media = audio

            def modern_chat(audio=False):
                prompt = "<|audio|>\nTranscribe the audio." if audio else image_prompt
                media = {"audios": [ov.Tensor(waveform)]} if audio else {"images": [ov.Tensor(pixels[None])]}
                history = genai.ChatHistory([{"role": "user", "content": prompt}])
                tokens, text = generate(history, **media)
                reference_prompt = "<__media__>\nTranscribe the audio." if audio else "<__media__>\nDescribe the image."
                reference_history = [{"role": "user", "content": reference_prompt}]
                reference_media = audio_media if audio else image
                first_case = compare(reference_history, tokens, text, "modern_audio" if audio else "modern_image",
                                     reference_media)
                # Switch to a different history with identical media-token geometry.
                # Token IDs alone cannot distinguish the two encoders' outputs.
                other = genai.ChatHistory([{"role": "user", "content": prompt}])
                other_media = ({"audios": [ov.Tensor(np.ascontiguousarray(waveform[::-1]))]} if audio else
                               {"images": [ov.Tensor((255 - pixels)[None])]})
                pipe.generate(other, max_new_tokens=1, do_sample=False, **other_media)
                history.append({"role": "assistant", "content": text})
                reference_history.append({"role": "assistant", "content": text})
                followup = "What did the speaker say?" if audio else "What is shown?"
                history.append({"role": "user", "content": followup})
                reference_history.append({"role": "user", "content": followup})
                case = compare(reference_history, *generate(history),
                               "modern_audio_chat" if audio else "modern_image_chat", reference_media)
                assert case_passed(first_case) and case_passed(case), (first_case, case)

            def beam_search():
                result = pipe.generate(image_prompt, images=[ov.Tensor(pixels[None])],
                    num_beams=3, num_return_sequences=2, max_new_tokens=8, min_new_tokens=8, do_sample=False)
                assert len(result.texts) == 2 and np.isfinite(result.scores).all()

            def cancel_and_reset():
                class Cancel(Tokens):
                    def write(self, tokens):
                        super().write(tokens)
                        return genai.StreamingStatus.CANCEL if len(self.tokens) >= 3 else genai.StreamingStatus.RUNNING
                pipe.generate(image_prompt, images=[ov.Tensor(pixels[None])], max_new_tokens=20, do_sample=False,
                              streamer=Cancel())
                tokens, _ = generate("What is 2 plus 2?")
                assert tokens == cases[0]["tokens"], tokens

            if args.chat:
                check("modern_image_chat", modern_chat)
                if args.audio:
                    check("modern_audio_chat", lambda: modern_chat(True))
            check("beam_search", beam_search)
            check("cancel_and_reset", cancel_and_reset)

        # A second request must start with an empty cache.
        reset_matches = generate("What is 2 plus 2?")[0] == cases[0]["tokens"]
    report.update(request_reset_matches=reset_matches, chat_reset_matches=chat_reset_matches)



if __name__ == "__main__":
    raise SystemExit(main())
