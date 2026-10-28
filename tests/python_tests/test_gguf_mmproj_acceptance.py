# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GGUF acceptance; fixtures and CPU oracle use pinned sources."""

import json
import os
import subprocess  # nosec B404
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from validate_gguf_mmproj import REFERENCE_REVISION

MODELS = {
    "gemma3": ("tiny-random-gemma3", "0cc4c5a822bbc68aeb312fd789046463ca301d14"),
    "gemma4": ("tiny-random-gemma4", "422d12880b72b92b1547ac7ce2e1a4865f6bec6c"),
    "muse": ("tiny-random-muse-glimmer", "cb70af191bbbf704a47b7d24dd5287d97dd60103"),
    "qwen35": ("tiny-random-qwen3.5", "aa8ac4a953c6081e0afaa34bd95cefd9c852ddbf"),
}
TEST_DIR = Path(__file__).parent


def execute(command, log, timeout=900):
    with log.open("w") as output:
        result = subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    assert result.returncode == 0, f"{command}: exit {result.returncode}\n{log.read_text()[-12000:]}"


@pytest.fixture(scope="session")
def mmproj_reference(tmp_path_factory):
    build = Path(os.environ.get("GGUF_LLAMA_BUILD_DIR") or tmp_path_factory.mktemp("mmproj_reference"))
    build.mkdir(parents=True, exist_ok=True)
    execute(
        [
            "cmake",
            "-S",
            TEST_DIR / "gguf_reference",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER_LAUNCHER=",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=",
        ],
        build / "configure-mmproj.log",
        timeout=300,
    )
    execute(
        ["cmake", "--build", build, "--target", "genai-gguf-mmproj-oracle", "genai-gguf-quantize", "--parallel", "4"],
        build / "build-mmproj.log",
    )
    return build / "_deps/llama_cpp-src", build / "bin"


def inventory(path):
    import hashlib
    from gguf import GGUFReader

    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensor_types": dict(Counter(t.tensor_type.name for t in GGUFReader(str(path)).tensors)),
    }


def materialize_muse_output(source, destination):
    from gguf import GGUFReader, GGUFWriter

    reader = GGUFReader(str(source))
    writer = GGUFWriter(str(destination), "muse-glimmer")
    for key, field in reader.fields.items():
        if field.types and key != "general.architecture" and not key.startswith("GGUF."):
            writer.add_key_value(
                key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None
            )
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data)
    if not any(t.name == "output.weight" for t in reader.tensors):
        writer.add_tensor("output.weight", next(t.data for t in reader.tensors if t.name == "token_embd.weight"))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture(scope="module", params=list(MODELS))
def mmproj_model(request, mmproj_reference, tmp_path_factory):
    family = request.param
    model, revision = MODELS[family]
    repo = "optimum-intel-internal-testing/" + model
    hf = Path(snapshot_download(repo, revision=revision))
    directory = tmp_path_factory.mktemp("mmproj_" + family)
    source, _ = mmproj_reference
    adaptation = None
    if family == "qwen35":
        fixture_hf = directory / "hf"
        fixture_hf.mkdir()
        config = json.loads((hf / "config.json").read_text())
        config["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 0]
        for file in hf.iterdir():
            if file.name == "config.json":
                (fixture_hf / file.name).write_text(json.dumps(config))
            else:
                (fixture_hf / file.name).symlink_to(file)
        hf = fixture_hf
        adaptation = "Tiny head_dim=16 uses explicit mrope_section=[1,1,0]; production defaults exceed its rotary width"
    language, mmproj = directory / "model-f32.gguf", directory / "mmproj-f32.gguf"
    for destination, options in ((language, ["--no-mtp"] if family == "qwen35" else []), (mmproj, ["--mmproj"])):
        execute(
            [
                sys.executable,
                source / "convert_hf_to_gguf.py",
                hf,
                "--outfile",
                destination,
                "--outtype",
                "f32",
                *options,
            ],
            destination.with_suffix(".log"),
        )
    manifest = {
        "repo": repo,
        "revision": revision,
        "converter_revision": REFERENCE_REVISION,
        "original_language": inventory(language),
        "fixture_adaptation": adaptation,
    }
    if family == "muse":
        repaired = directory / "model-tied-output-f32.gguf"
        materialize_muse_output(language, repaired)
        language = repaired
        manifest["fixture_adaptation"] = "Materialize output.weight from tied token_embd.weight for the CPU oracle"
    manifest.update(language=inventory(language), mmproj=inventory(mmproj))
    provenance = directory / "sources.json"
    provenance.write_text(json.dumps(manifest, indent=2))
    return family, hf, language, mmproj, provenance


@pytest.fixture(scope="session")
def mmproj_media(mmproj_reference, tmp_path_factory):
    source, _ = mmproj_reference
    audio = tmp_path_factory.mktemp("mmproj_media") / "audio.wav"
    execute(
        [
            "ffmpeg",
            "-y",
            "-i",
            source / "tools/mtmd/test-2.mp3",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            audio,
        ],
        audio.with_suffix(".log"),
    )
    return source / "tools/mtmd/test-1.jpeg", audio


def acceptance(family, language, mmproj, provenance, backend, image, audio, reference, tmp_path, q4=False):
    _, binaries = reference
    report_dir = Path(os.environ.get("GGUF_MMPROJ_REPORT_DIR", tmp_path))
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"{family}-{'q4_0' if q4 else 'f32'}-{backend}-{'jpeg' if image else 'square'}"
    command = [
        sys.executable,
        TEST_DIR / "validate_gguf_mmproj.py",
        language,
        mmproj,
        "--oracle",
        binaries / "genai-gguf-mmproj-oracle",
        "--family",
        family,
        "--attention-backend",
        backend,
        "--full-coverage",
        "--multi-media",
        "--chat",
        "--api-checks",
        "--reference-manifest",
        provenance,
        "--report",
        report_dir / (name + ".json"),
    ]
    if image:
        command += ["--image", image]
    if family != "gemma3":
        command += ["--video-frames", "4"]
    if family == "gemma4":
        command += ["--audio", audio, "--audio-boundaries"]
    if q4:
        command += ["--require-q4-0"]
    else:
        command += ["--reference-kind", "represented-f32"]
    if os.environ.get("GGUF_RUNTIME_MANIFEST"):
        command += ["--runtime-manifest", os.environ["GGUF_RUNTIME_MANIFEST"]]
    execute(command, report_dir / (name + ".log"), timeout=1800)


@pytest.mark.parametrize("backend", ["SDPA", "PA"])
@pytest.mark.parametrize("real_image", [False, True], ids=["square", "jpeg"])
def test_mmproj_llama_acceptance(mmproj_model, mmproj_media, mmproj_reference, backend, real_image, tmp_path):
    family, _, language, mmproj, provenance = mmproj_model
    image, audio = mmproj_media
    acceptance(
        family, language, mmproj, provenance, backend, image if real_image else None, audio, mmproj_reference, tmp_path
    )


@pytest.fixture(scope="module")
def q4_mmproj_model(mmproj_reference, tmp_path_factory):
    directory = tmp_path_factory.mktemp("mmproj_q4_0")
    repo, revision = "ggml-org/Qwen3.5-0.8B-GGUF", "8fea620810c4afa23dd6443f999a48574c1611a3"
    original = Path(hf_hub_download(repo, "Qwen3.5-0.8B-Q4_0.gguf", revision=revision))
    mmproj = Path(
        hf_hub_download(
            "unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf", revision="6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
        )
    )
    expanded, language = directory / "represented-f16.gguf", directory / "pure-q4_0.gguf"
    _, binaries = mmproj_reference
    for input_path, output_path, precision in ((original, expanded, "F16"), (expanded, language, "Q4_0")):
        execute([binaries / "genai-gguf-quantize", input_path, output_path, precision], output_path.with_suffix(".log"))
    provenance = directory / "sources.json"
    provenance.write_text(
        json.dumps(
            {
                "repo": repo,
                "revision": revision,
                "quantizer_revision": REFERENCE_REVISION,
                "mmproj_repo": "unsloth/Qwen3.5-0.8B-GGUF",
                "mmproj_revision": "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
                "fixture_adaptation": (
                    "F16 expansion then pure Q4_0 creates a new fixture; " "original mixed-weight parity is not claimed"
                ),
                "original": inventory(original),
                "expanded": inventory(expanded),
                "language": inventory(language),
                "mmproj": inventory(mmproj),
            },
            indent=2,
        )
    )
    return language, mmproj, provenance


@pytest.mark.parametrize("backend", ["SDPA", "PA"])
def test_mmproj_q4_0_llama_acceptance(q4_mmproj_model, mmproj_media, mmproj_reference, backend, tmp_path):
    language, mmproj, provenance = q4_mmproj_model
    image, audio = mmproj_media
    acceptance("qwen35", language, mmproj, provenance, backend, image, audio, mmproj_reference, tmp_path, q4=True)


def test_mmproj_optimum_directory_and_map(mmproj_model, tmp_path):
    family, hf, _, _, _ = mmproj_model
    export = tmp_path / "ov"
    execute(
        [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            hf,
            "--task",
            "image-text-to-text",
            "--weight-format",
            "fp32",
            export,
        ],
        tmp_path / "export.log",
    )
    models = {}
    for xml in export.glob("openvino_*_model.xml"):
        models[xml.stem[len("openvino_") : -len("_model")]] = (
            xml.read_text(),
            ov.Tensor(np.fromfile(xml.with_suffix(".bin"), np.uint8)),
        )
    assert models, f"{family}: no exported models"
    properties = dict(INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_NUM_THREADS=4)
    directory = genai.VLMPipeline(str(export), "CPU", **properties)
    serialized = genai.VLMPipeline(models, genai.Tokenizer(str(export)), str(export), "CPU", **properties)
    for prompt, media in (
        ("What is 2 plus 2?", {}),
        ("Describe the image.", {"images": [ov.Tensor(np.zeros((1, 64, 80, 3), np.uint8))]}),
    ):
        assert (
            directory.generate(prompt, max_new_tokens=5, do_sample=False, **media).texts
            == serialized.generate(prompt, max_new_tokens=5, do_sample=False, **media).texts
        )
