#!/usr/bin/env python3
"""Transcribes wav files with Voxtral Realtime, running both the audio encoder and the text decoder on PagedAttention.

Prototype of the approach in pure OpenVINO Python API (openvino_genai is not used):

    log-mel features -> audio_conv -> audio_transformer (PagedAttention, sliding window)
                     -> audio_adapter -> + text_embeddings -> text_decoder (PagedAttention, sliding window)

Audio is processed the way the streaming pipeline does it: a prefill, then one text position per 80 ms of audio.
Several wav files are transcribed concurrently, batched in every model.

Usage:
    .venv/bin/python voxtral_pa_transcribe.py audio.wav
    .venv/bin/python voxtral_pa_transcribe.py first.wav second.wav --model-dir /path/to/voxtral-stateful

MODEL_DIR must hold audio_conv/, audio_transformer/, audio_adapter/, text_embeddings/ and text_decoder/ next to
config.json, preprocessor_config.json and tekken.json.
"""

import argparse
import base64
import codecs
import contextlib
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import openvino as ov
from openvino import opset13 as ops
from openvino._offline_transformations import paged_attention_transformation

DEFAULT_MODEL_DIR = Path.home() / "models" / "OpenVINO" / "voxtral-stateful"
MODEL_FILE = "openvino_model.xml"
MODEL_PARTS = ("audio_conv", "audio_transformer", "audio_adapter", "text_embeddings", "text_decoder")
STREAMING_PAD_TOKEN = "[STREAMING_PAD]"
# PagedAttentionExtension input that holds the sliding window size; SDPAToPagedAttention leaves it at 0 (no window).
SLIDING_WINDOW_INPUT_INDEX = 10
VALUE_CACHE_INPUT_PREFIX = "value_cache."
KV_CACHE_INPUT_PREFIXES = ("key_cache.", VALUE_CACHE_INPUT_PREFIX)
# Cache inputs are [blocks, heads, block size, head size].
BLOCK_SIZE_DIMENSION = 2
CONV1_CACHE_FRAMES = 2
CONV2_CACHE_FRAMES = 1
# Whisper-style log-mel normalization.
LOG_MEL_FLOOR = 1e-10
LOG_MEL_DYNAMIC_RANGE = 8.0
LOG_MEL_OFFSET = 4.0
LOG_MEL_SCALE = 4.0
MS_PER_SECOND = 1e3


class Settings:
    """Values the C++ Voxtral pipeline derives from config.json, tekken.json and preprocessor_config.json."""

    def __init__(self, model_dir):
        config = json.loads((model_dir / "config.json").read_text())
        tekken = json.loads((model_dir / "tekken.json").read_text())
        preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text())
        audio, text = config["audio_config"], config["text_config"]

        self.audio_window = audio["sliding_window"]
        self.audio_hidden_size = audio["hidden_size"]
        self.text_window = text["sliding_window"]
        self.bos_token_id = text["bos_token_id"]
        self.eos_token_id = text["eos_token_id"]
        self.frames_per_position = config["audio_length_per_tok"]
        self.audio_tokens_per_position = config["downsample_factor"]
        self.num_delay_tokens = config["default_num_delay_tokens"]
        self.left_pad_tokens = tekken["audio"]["streaming_n_left_pad_tokens"]
        self.num_special_tokens = tekken["config"]["default_num_special_tokens"]
        self.streaming_pad_token_id = next(
            token["rank"] for token in tekken["special_tokens"] if token["token_str"] == STREAMING_PAD_TOKEN
        )
        self.pieces = {entry["rank"]: base64.b64decode(entry["token_bytes"]) for entry in tekken["vocab"]}

        self.num_mel_bins = preprocessor["feature_size"]
        self.sampling_rate = preprocessor["sampling_rate"]
        self.n_fft = preprocessor["n_fft"]
        self.hop_length = preprocessor["hop_length"]
        self.global_log_mel_max = preprocessor["global_log_mel_max"]

        self.prefill_positions = 1 + self.left_pad_tokens + self.num_delay_tokens
        self.left_pad_samples = self.left_pad_tokens * self.frames_per_position * self.hop_length
        self.step_samples = self.frames_per_position * self.hop_length
        # Audio the prefill needs: the frames of the first delayed position, as a centered STFT.
        self.first_chunk_samples = (self.num_delay_tokens + 1) * self.step_samples - self.hop_length + self.n_fft // 2

    def count_steps(self, num_samples):
        """Text positions decoded after the prefill, as VoxtralASRStream::process_audio() and flush() produce them."""
        overlap = self.n_fft - self.hop_length
        remaining = max(num_samples, self.first_chunk_samples) - (self.first_chunk_samples - overlap)
        steps = 0
        while remaining >= self.step_samples + self.n_fft:
            steps += 1
            remaining -= self.step_samples
            overlap = self.n_fft
        if remaining > overlap:
            steps += 1
        return steps + self.num_delay_tokens

    def decode_token(self, token_id):
        if token_id < self.num_special_tokens:
            return b""
        return self.pieces[token_id - self.num_special_tokens]


class LogMel:
    """Whisper-style log-mel frames; frame i is centered on sample i * hop_length."""

    def __init__(self, settings):
        self.n_fft = settings.n_fft
        self.hop_length = settings.hop_length
        # Periodic Hann window.
        self.window = np.hanning(self.n_fft + 1)[:-1]
        self.filters = librosa.filters.mel(sr=settings.sampling_rate, n_fft=self.n_fft, n_mels=settings.num_mel_bins)
        self.floor = settings.global_log_mel_max - LOG_MEL_DYNAMIC_RANGE

    def padded_length(self, num_frames):
        return (num_frames - 1) * self.hop_length + self.n_fft

    def pad(self, waveform, num_frames):
        """Zero-pads so that num_frames centered frames can be taken; the waveform starts half a window in."""
        padded = np.zeros(self.padded_length(num_frames), dtype=np.float32)
        begin = self.n_fft // 2
        usable = min(len(waveform), len(padded) - begin)
        padded[begin : begin + usable] = waveform[:usable]
        return padded

    def frames(self, padded, first_frame, num_frames):
        """[num_mel_bins, num_frames] features of frames first_frame.. of a pad() result."""
        begin = first_frame * self.hop_length
        samples = padded[begin : begin + self.padded_length(num_frames)]
        windows = np.lib.stride_tricks.sliding_window_view(samples, self.n_fft)[:: self.hop_length]
        power = np.abs(np.fft.rfft(windows * self.window, axis=1)) ** 2
        log_mel = np.log10(np.maximum(power @ self.filters.T, LOG_MEL_FLOOR))
        return ((np.maximum(log_mel, self.floor) + LOG_MEL_OFFSET) / LOG_MEL_SCALE).T.astype(np.float32)


class PagedModel:
    """A stateful transformer converted to PagedAttention and shared by several sequences.

    Each sequence owns a list of KV-cache blocks. Blocks whose tokens left the sliding window go back to the pool.
    Positions keep growing, so nothing cached is ever re-rotated.
    """

    def __init__(self, core, model_path, device, window, sequence_lengths, longest_step, constant_inputs=None):
        model = core.read_model(model_path)
        paged_attention_transformation(model)
        for node in model.get_ops():
            if node.get_type_name() != "PagedAttentionExtension":
                continue
            current = node.input(SLIDING_WINDOW_INPUT_INDEX).get_source_output()
            restored = ops.constant(np.array(window, dtype=current.get_element_type().to_dtype()))
            node.input(SLIDING_WINDOW_INPUT_INDEX).replace_source_output(restored.output(0))
        compiled = core.compile_model(model, device)
        cache_inputs = [port for port in compiled.inputs if port.get_any_name().startswith(KV_CACHE_INPUT_PREFIXES)]
        # Quantized key caches hold extra rows per block, so only a value cache tells the block size.
        value_cache = next(port for port in cache_inputs if port.get_any_name().startswith(VALUE_CACHE_INPUT_PREFIX))
        self.block_size = value_cache.get_partial_shape()[BLOCK_SIZE_DIMENSION].get_length()
        self.window = window
        # A sequence never caches more than its window plus one partly evictable block plus the tokens of a step.
        most_tokens = window + self.block_size + longest_step
        num_blocks = sum(-(-min(length, most_tokens) // self.block_size) for length in sequence_lengths)
        self.request = compiled.create_infer_request()
        for port in cache_inputs:
            shape = [num_blocks] + [dimension.get_length() for dimension in list(port.get_partial_shape())[1:]]
            self.request.set_tensor(port.get_any_name(), ov.Tensor(port.get_element_type(), shape))
        for name, value in (constant_inputs or {}).items():
            self.request.set_tensor(name, ov.Tensor(value))
        self.free_blocks = list(range(num_blocks))
        self.blocks = [[] for _ in sequence_lengths]
        self.cached = [0] * len(sequence_lengths)
        self.positions = [0] * len(sequence_lengths)
        self.output_size = compiled.output(0).get_partial_shape()[-1].get_length()

    def infer(self, sequences, embeddings):
        """Appends embeddings[i] ([tokens, hidden]) to sequence sequences[i]; returns [tokens, output] per sequence."""
        past_lens, begins, block_begins, block_indices, position_ids = [], [0], [0], [], []
        for sequence, embeds in zip(sequences, embeddings):
            blocks = self.blocks[sequence]
            # The first new token still attends to the last window - 1 cached ones.
            visible = min(self.cached[sequence], self.window - 1)
            for _ in range((self.cached[sequence] - visible) // self.block_size):
                self.free_blocks.append(blocks.pop(0))
                self.cached[sequence] -= self.block_size
            while len(blocks) * self.block_size < self.cached[sequence] + len(embeds):
                blocks.append(self.free_blocks.pop())
            past_lens.append(self.cached[sequence])
            begins.append(begins[-1] + len(embeds))
            block_indices.extend(blocks)
            block_begins.append(len(block_indices))
            position_ids.append(np.arange(self.positions[sequence], self.positions[sequence] + len(embeds)))
            self.cached[sequence] += len(embeds)
            self.positions[sequence] += len(embeds)
        request = self.request
        request.set_tensor("inputs_embeds", ov.Tensor(np.ascontiguousarray(np.concatenate(embeddings), np.float32)))
        request.set_tensor("position_ids", ov.Tensor(np.concatenate(position_ids).astype(np.int64)))
        request.set_tensor("past_lens", ov.Tensor(np.array(past_lens, np.int32)))
        request.set_tensor("subsequence_begins", ov.Tensor(np.array(begins, np.int32)))
        request.set_tensor("block_indices", ov.Tensor(np.array(block_indices, np.int32)))
        request.set_tensor("block_indices_begins", ov.Tensor(np.array(block_begins, np.int32)))
        request.set_tensor("max_context_len", ov.Tensor(np.array(max(self.cached[s] for s in sequences), np.int32)))
        request.infer()
        output = request.get_output_tensor().data.reshape(-1, self.output_size)
        return [output[begin:end] for begin, end in zip(begins, begins[1:])]


class Stream:
    """One wav file being transcribed."""

    def __init__(self, path, settings, log_mel):
        self.path = path
        audio = librosa.load(path, sr=settings.sampling_rate, mono=True)[0].astype(np.float32)
        self.duration = len(audio) / settings.sampling_rate
        # Empty audio produces no text position at all.
        self.num_positions = settings.prefill_positions + settings.count_steps(len(audio)) if len(audio) else 0
        left_padded = np.concatenate([np.zeros(settings.left_pad_samples, np.float32), audio])
        self.waveform = log_mel.pad(left_padded, self.num_positions * settings.frames_per_position)
        self.positions_done = 0
        # Index of this stream in the PagedAttention models; set when transcription starts.
        self.sequence = None
        self.conv1_cache = np.zeros((settings.num_mel_bins, CONV1_CACHE_FRAMES), np.float32)
        self.conv2_cache = np.zeros((settings.audio_hidden_size, CONV2_CACHE_FRAMES), np.float32)
        self.last_token = None
        self.finished = self.num_positions == 0
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""

    def accept(self, token_id, positions, settings):
        """Records the token decoded after `positions` more text positions; returns the newly completed text."""
        self.last_token = token_id
        self.positions_done += positions
        is_stop_token = token_id == settings.eos_token_id
        self.finished = is_stop_token or self.positions_done == self.num_positions
        piece = b"" if is_stop_token else settings.decode_token(token_id)
        emitted = self.utf8.decode(piece, final=self.finished)
        self.text += emitted
        return emitted


class StageTimer:
    def __init__(self):
        self.seconds = {}

    @contextlib.contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        yield
        self.seconds[stage] = self.seconds.get(stage, 0.0) + time.perf_counter() - start


def transcribe(streams, settings, log_mel, core, model_dir, device, timer, on_text):
    """Runs all streams to the end, one batched step per text position. Returns the number of steps."""
    sequences = [stream for stream in streams if not stream.finished]
    for index, stream in enumerate(sequences):
        stream.sequence = index
    if not sequences:
        return 0

    def compile_part(name):
        return core.compile_model(model_dir / name / MODEL_FILE, device).create_infer_request()

    with timer.measure("load"):
        audio_conv = compile_part("audio_conv")
        audio_adapter = compile_part("audio_adapter")
        text_embeddings = compile_part("text_embeddings")
        audio_transformer = PagedModel(
            core,
            model_dir / "audio_transformer" / MODEL_FILE,
            device,
            settings.audio_window,
            [stream.num_positions * settings.audio_tokens_per_position for stream in sequences],
            settings.prefill_positions * settings.audio_tokens_per_position,
        )
        text_decoder = PagedModel(
            core,
            model_dir / "text_decoder" / MODEL_FILE,
            device,
            settings.text_window,
            [stream.num_positions for stream in sequences],
            settings.prefill_positions,
            {"num_delay_tokens": np.array([settings.num_delay_tokens], np.float32)},
        )

    prompt = [settings.bos_token_id] + [settings.streaming_pad_token_id] * (settings.prefill_positions - 1)
    steps = 0
    active = sequences
    while active:
        # Every stream starts with the prefill, so all active streams are at the same kind of step.
        is_prefill = steps == 0
        positions = settings.prefill_positions if is_prefill else 1
        num_frames = positions * settings.frames_per_position
        ids = [stream.sequence for stream in active]

        with timer.measure("features"):
            features = np.stack(
                [
                    log_mel.frames(stream.waveform, stream.positions_done * settings.frames_per_position, num_frames)
                    for stream in active
                ]
            )
        with timer.measure("audio conv"):
            audio_conv.set_tensor("input_features", ov.Tensor(features))
            audio_conv.set_tensor("conv1_cache", ov.Tensor(np.stack([stream.conv1_cache for stream in active])))
            audio_conv.set_tensor("conv2_cache", ov.Tensor(np.stack([stream.conv2_cache for stream in active])))
            audio_conv.infer()
            conv_embeddings = audio_conv.get_tensor("inputs_embeds").data
            for row, stream in enumerate(active):
                stream.conv1_cache = audio_conv.get_tensor("conv1_cache_1").data[row].copy()
                stream.conv2_cache = audio_conv.get_tensor("conv2_cache_1").data[row].copy()
        with timer.measure("audio transformer (PA)"):
            hidden = np.stack(audio_transformer.infer(ids, list(conv_embeddings)))
        with timer.measure("audio adapter"):
            audio_adapter.set_tensor("hidden_states", ov.Tensor(hidden))
            audio_adapter.infer()
            audio_embeddings = audio_adapter.get_output_tensor().data
        with timer.measure("text embeddings"):
            input_ids = [prompt if is_prefill else [stream.last_token] for stream in active]
            text_embeddings.set_input_tensor(ov.Tensor(np.array(input_ids, np.int64)))
            text_embeddings.infer()
            inputs_embeds = text_embeddings.get_output_tensor().data + audio_embeddings
        with timer.measure("text decoder (PA)"):
            logits = text_decoder.infer(ids, list(inputs_embeds))
            tokens = [int(rows[-1].argmax()) for rows in logits]

        for stream, token in zip(active, tokens):
            on_text(stream, stream.accept(token, positions, settings))
        active = [stream for stream in active if not stream.finished]
        steps += 1
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wav", nargs="+", type=Path, help="Audio files; several are transcribed concurrently.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="CPU", help="PagedAttention is supported on CPU.")
    args = parser.parse_args()

    missing = [
        str(args.model_dir / part / MODEL_FILE)
        for part in MODEL_PARTS
        if not (args.model_dir / part / MODEL_FILE).exists()
    ]
    if missing:
        sys.exit("Missing model files:\n  " + "\n  ".join(missing))

    settings = Settings(args.model_dir)
    log_mel = LogMel(settings)
    streams = [Stream(path, settings, log_mel) for path in args.wav]
    live = len(streams) == 1

    def on_text(stream, text):
        if live:
            print(text, end="", flush=True)

    timer = StageTimer()
    wall_start = time.perf_counter()
    steps = transcribe(streams, settings, log_mel, ov.Core(), args.model_dir, args.device, timer, on_text)
    load_seconds = timer.seconds.pop("load", 0.0)
    inference_seconds = time.perf_counter() - wall_start - load_seconds

    if live:
        print()
    else:
        for stream in streams:
            print(f"[{stream.path.name}] {stream.text.strip()}")

    longest = max(stream.duration for stream in streams)
    step_ms = settings.frames_per_position * settings.hop_length / settings.sampling_rate * MS_PER_SECOND
    print(
        f"\n{len(streams)} stream(s), longest audio {longest:.1f} s, model load {load_seconds:.1f} s", file=sys.stderr
    )
    if not steps:
        return
    print(
        f"transcription {inference_seconds:.1f} s = {inference_seconds / longest:.2f} s per 1 s of audio "
        f"({steps} steps, {step_ms:.0f} ms of audio each)",
        file=sys.stderr,
    )
    for stage, seconds in timer.seconds.items():
        print(f"  {stage:24s} {seconds * MS_PER_SECOND / steps:7.1f} ms per step", file=sys.stderr)


if __name__ == "__main__":
    main()
