# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import shutil
import subprocess  # nosec B404
from pathlib import Path

import numpy as np
import pytest
import torch

from .atomic_download import AtomicDownloadManager
from .network import retry_request

logger = logging.getLogger(__name__)

KOKORO_STYLE_DIM = 128
KOKORO_REF_S_DIM = 256
KOKORO_VOICE_PACK_LENGTHS = 510
KOKORO_TINY_MODEL_SEED = 1337
KOKORO_MAX_DUR = 10
KOKORO_EXPORT_TIMEOUT_SECONDS = 300

TINY_G2P_SEED = 42
TINY_G2P_EXPORT_TIMEOUT_SECONDS = 120

_G2P_GRAPHEME_CHARS = "____AIOWYbdfhijklmnpstuvwz'-.BCDEFGHJKLMNPQRSTUVXZacegoqrxy"
_G2P_PHONEME_CHARS = (
    "____AIOWYbdfhijklmnpstuvwz"
    "\u00e6\u00f0\u014b\u0251\u0254\u0259\u025b\u025c\u0261\u026a"
    "\u0279\u027e\u0283\u028a\u028c\u0292\u0294\u02a4\u02a7\u02c8"
    "\u02cc\u03b8\u1d4a\u1d7b"
)
_G2P_VOCAB_SIZE = 63


def generate_speaker_embedding(shape: tuple[int, ...]) -> np.ndarray:
    """Generate a deterministic speaker embedding/voice-pack for testing."""
    style_dim = shape[-1]
    base = np.linspace(-0.25, 0.25, num=style_dim, dtype=np.float32)
    base = base / (np.linalg.norm(base) + 1e-8)

    if len(shape) == 3:
        return np.broadcast_to(base.reshape(1, 1, style_dim), shape).copy()
    if len(shape) == 2:
        return np.broadcast_to(base.reshape(1, style_dim), shape).copy()
    if len(shape) == 1:
        return base.copy()

    raise ValueError(f"Unsupported speaker embedding shape: {shape}")


def _flatten_atomic_model_dir(model_path: Path) -> None:
    # AtomicDownloadManager moves the temp root directory into model_path.
    # Our creators emit files under temp/model, so flatten that one level.
    nested = model_path / "model"
    if nested.exists() and (nested / "config.json").exists():
        if not (model_path / "config.json").exists():
            for item in nested.iterdir():
                shutil.move(str(item), str(model_path / item.name))
        shutil.rmtree(nested, ignore_errors=True)


def generate_tiny_kokoro_model(output_dir: Path) -> Path:
    """Generate a tiny random Kokoro checkpoint in local HF layout."""
    try:
        from kokoro.istftnet import Decoder
        from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
        from transformers import AlbertConfig
    except ImportError as error:
        pytest.skip(f"Required Kokoro dependencies not available: {error}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(KOKORO_TINY_MODEL_SEED)
    np.random.seed(KOKORO_TINY_MODEL_SEED)
    torch.manual_seed(KOKORO_TINY_MODEL_SEED)

    symbols = (
        ";:,.!?-—()'\"/ "
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "əɚɝɪʊʌæɑɔɛɜɒɹɾθðŋʃʒʤʧˈˌ"
        "àáâãäåèéêëìíîïòóôõöùúûüýÿ"
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for ch in symbols:
        if ch not in seen:
            deduped.append(ch)
            seen.add(ch)
        if len(deduped) >= 177:
            break
    vocab = {ch: i + 1 for i, ch in enumerate(deduped)}

    config = {
        "model_type": "kokoro",
        "export_model_type": "kokoro",
        "hidden_dim": 512,
        "style_dim": KOKORO_STYLE_DIM,
        "n_token": 178,
        "n_layer": 1,
        "dim_in": 512,
        "n_mels": 80,
        "max_dur": KOKORO_MAX_DUR,
        "dropout": 0.2,
        "text_encoder_kernel_size": 3,
        "plbert": {
            "hidden_size": 128,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "num_hidden_layers": 2,
            "dropout": 0.1,
        },
        "istftnet": {
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
        },
        "vocab": vocab,
        "multispeaker": True,
        "max_conv_dim": 512,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    with torch.no_grad():
        bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        bert_encoder = torch.nn.Linear(config["plbert"]["hidden_size"], config["hidden_dim"])
        predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            **config["istftnet"],
        )

        torch.save(
            {
                "bert": bert.state_dict(),
                "bert_encoder": bert_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "decoder": decoder.state_dict(),
            },
            output_dir / "tiny-kokoro-random.pth",
        )

    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_pack = torch.from_numpy(generate_speaker_embedding((KOKORO_VOICE_PACK_LENGTHS, 1, KOKORO_REF_S_DIM)))
    torch.save(voice_pack, voices_dir / "tiny_voice.pt")

    logger.info("Generated tiny Kokoro model at %s", output_dir)
    return output_dir


def export_kokoro_model_to_openvino(local_model_path: Path, output_path: Path) -> Path:
    """Export a local Kokoro model to OpenVINO IR via optimum-cli."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        "optimum-cli",
        "export",
        "openvino",
        "-m",
        str(local_model_path),
        str(output_path),
        "--trust-remote-code",
    ]

    try:
        retry_request(
            lambda: subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=KOKORO_EXPORT_TIMEOUT_SECONDS,
            )
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Kokoro export timed out after {KOKORO_EXPORT_TIMEOUT_SECONDS} seconds\n"
            f"stdout: {error.stdout or ''}\nstderr: {error.stderr or ''}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Kokoro export failed with return code {error.returncode}\n"
            f"stdout: {error.stdout or ''}\nstderr: {error.stderr or ''}"
        )

    logger.info("Successfully exported tiny Kokoro model to %s", output_path)
    return output_path


def ensure_tiny_voice_bin(ov_path: Path, tiny_kokoro_model_path: Path) -> Path:
    """Ensure tiny_voice.bin exists under OV export using tiny_voice.pt fallback."""
    voices_dir = ov_path / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    voice_bin = voices_dir / "tiny_voice.bin"
    if voice_bin.exists():
        return voice_bin

    source_pt = tiny_kokoro_model_path / "voices" / "tiny_voice.pt"
    if not source_pt.exists():
        raise FileNotFoundError(f"Missing source tiny voice pack: {source_pt}")

    source_voice = torch.load(source_pt, map_location="cpu")
    if isinstance(source_voice, torch.Tensor):
        voice_np = source_voice.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        voice_np = np.asarray(source_voice, dtype=np.float32)

    voice_np.tofile(voice_bin)
    logger.info("Created fallback tiny voice pack at %s", voice_bin)
    return voice_bin


def generate_tiny_g2p_model(output_dir: Path) -> Path:
    """Generate a tiny random BART G2P model with Kokoro fallback vocab fields."""
    try:
        from transformers import BartConfig, BartForConditionalGeneration
    except ImportError as error:
        pytest.skip(f"Required transformers dependency not available: {error}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(TINY_G2P_SEED)
    np.random.seed(TINY_G2P_SEED)
    torch.manual_seed(TINY_G2P_SEED)

    config = BartConfig(
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        vocab_size=_G2P_VOCAB_SIZE,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        pad_token_id=0,
        forced_eos_token_id=2,
        scale_embedding=False,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="gelu",
        use_cache=True,
    )

    config.grapheme_chars = _G2P_GRAPHEME_CHARS
    config.phoneme_chars = _G2P_PHONEME_CHARS

    model = BartForConditionalGeneration(config)
    model.eval()

    # Use PyTorch .bin instead of safetensors here. For some reason,
    # safetensors serialization can fail in CI trying to write to the
    # mounted cache path.
    config.save_pretrained(str(output_dir))
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 1,
        "decoder_start_token_id": 1,
        "eos_token_id": 2,
        "forced_eos_token_id": 2,
        "pad_token_id": 0,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as handle:
        json.dump(generation_config, handle, indent=2)

    logger.info("Generated tiny G2P model at %s", output_dir)
    return output_dir


def export_g2p_model_to_openvino(local_model_path: Path, output_path: Path) -> Path:
    """Export tiny G2P model to OpenVINO encoder/decoder IR files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        "optimum-cli",
        "export",
        "openvino",
        "-m",
        str(local_model_path),
        str(output_path),
        "--task",
        "text2text-generation",
    ]

    try:
        retry_request(
            lambda: subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=TINY_G2P_EXPORT_TIMEOUT_SECONDS,
            )
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"G2P export timed out after {TINY_G2P_EXPORT_TIMEOUT_SECONDS} seconds\n"
            f"stdout: {error.stdout or ''}\nstderr: {error.stderr or ''}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"G2P export failed with return code {error.returncode}\n"
            f"stdout: {error.stdout or ''}\nstderr: {error.stderr or ''}"
        )

    logger.info("Successfully exported tiny G2P model to %s", output_path)
    return output_path


def prepare_tiny_kokoro_model_path(converted_models_dir: Path) -> Path:
    """Prepare tiny random Kokoro model in cache and return its path."""
    model_path = Path(converted_models_dir) / "tiny-random-kokoro"
    manager = AtomicDownloadManager(model_path)
    manager.execute(lambda temp_path: generate_tiny_kokoro_model(temp_path / "model"))
    _flatten_atomic_model_dir(model_path)
    return model_path


def prepare_tiny_kokoro_ov_path(converted_models_dir: Path, tiny_kokoro_model_path: Path) -> Path:
    """Prepare tiny random Kokoro OpenVINO export in cache and return its path."""
    ov_path = Path(converted_models_dir) / "tiny-random-kokoro-ov"
    voice_bin = ov_path / "voices" / "tiny_voice.bin"

    if not (ov_path / "openvino_model.xml").exists() or not voice_bin.exists():
        if ov_path.exists():
            shutil.rmtree(ov_path, ignore_errors=True)
        export_kokoro_model_to_openvino(tiny_kokoro_model_path, ov_path)

    ensure_tiny_voice_bin(ov_path, tiny_kokoro_model_path)
    return ov_path


def prepare_tiny_g2p_model_path(converted_models_dir: Path) -> Path:
    """Prepare tiny random BART G2P model in cache and return its path."""
    model_path = Path(converted_models_dir) / "tiny-random-g2p"
    manager = AtomicDownloadManager(model_path)
    manager.execute(lambda temp_path: generate_tiny_g2p_model(temp_path / "model"))
    _flatten_atomic_model_dir(model_path)
    return model_path


def prepare_tiny_g2p_ov_path(converted_models_dir: Path, tiny_g2p_model_path: Path) -> Path:
    """Prepare tiny random G2P OpenVINO export in cache and return its path."""
    ov_path = Path(converted_models_dir) / "tiny-random-g2p-ov"
    encoder_xml = ov_path / "openvino_encoder_model.xml"
    decoder_xml = ov_path / "openvino_decoder_model.xml"

    if not encoder_xml.exists() or not decoder_xml.exists():
        if ov_path.exists():
            shutil.rmtree(ov_path, ignore_errors=True)
        export_g2p_model_to_openvino(tiny_g2p_model_path, ov_path)

    return ov_path
