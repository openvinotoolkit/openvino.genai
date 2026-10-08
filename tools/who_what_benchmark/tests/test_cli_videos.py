import sys
import pytest
import torch
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
            "--taylorseer-config",
            '{"disable_cache_after_step": 0}',
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
            "--target-data",
            tmp_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai_with_taylorseer(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    # Test with full TaylorSeer config
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
            "--taylorseer-config",
            '{"cache_interval": 3, "disable_cache_before_step": 4, "disable_cache_after_step": -1}',
        ]
    )

    assert "Metrics for model" in output
    assert "TaylorSeer config:" in output
    similarity = get_similarity(output)
    assert similarity >= 0.97


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "image-to-video")],
)
def test_image2video_model_genai(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--hf",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
            "--target-data",
            tmp_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )


def run_test_with_lora(model_id, model_type, tmp_path, *, genai_threshold):
    if sys.platform == "darwin":
        pytest.xfail("Not enough memory on macOS CI runners. Ticket CVS-179749")
    if sys.platform == "win32":
        pytest.xfail("Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")

    from ov_utils import get_ov_cache_dir, download_hf_files_to_cache

    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(model_id)

    lora_cache_dir = get_ov_cache_dir() / "test_data" / "ltx_tiny_dummy_lora"
    lora_dir = download_hf_files_to_cache(
        "goyaladitya05/openvino-genai-test-files",
        lora_cache_dir,
        ["ltx_tiny_dummy_lora.safetensors"],
    )
    lora_file = lora_dir / "ltx_tiny_dummy_lora.safetensors"
    assert lora_file.exists(), f"LoRA adapter wasn't downloaded: {lora_file}"

    # 1) Generate GT using HF + LoRA
    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--hf",
        ]
    )
    assert gt_file.exists(), f"GT wasn't generated: {gt_file}"
    assert (tmp_path / "reference").exists()

    # 2) Target: GenAI + LoRA
    outputs_genai = tmp_path / "genai_lora"
    out_genai = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--output",
            outputs_genai,
        ]
    )

    assert "Metrics for model" in out_genai
    assert (outputs_genai / "target").exists()
    similarity = get_similarity(out_genai)
    assert similarity >= genai_threshold

    # 3) GenAI + LoRA at load time, empty AdapterConfig at generate time
    outputs_empty = tmp_path / "genai_empty_adapters"
    out_empty = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--empty_adapters",
            "--output",
            outputs_empty,
        ]
    )
    assert "Metrics for model" in out_empty
    assert (outputs_empty / "target").exists()


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai_with_lora(model_id, model_type, tmp_path):
    run_test_with_lora(model_id, model_type, tmp_path, genai_threshold=0.88)


# diffusers names of LTX-2 tensors -> original names, per dot-separated part of the name
LTX2_TRANSFORMER_TO_ORIGINAL = {
    "proj_in": "patchify_proj",
    "audio_proj_in": "audio_patchify_proj",
    "time_embed": "adaln_single",
    "audio_time_embed": "audio_adaln_single",
    "prompt_adaln": "prompt_adaln_single",
    "audio_prompt_adaln": "audio_prompt_adaln_single",
    "av_cross_attn_video_scale_shift": "av_ca_video_scale_shift_adaln_single",
    "av_cross_attn_audio_scale_shift": "av_ca_audio_scale_shift_adaln_single",
    "av_cross_attn_video_a2v_gate": "av_ca_a2v_gate_adaln_single",
    "av_cross_attn_audio_v2a_gate": "av_ca_v2a_gate_adaln_single",
    "video_a2v_cross_attn_scale_shift_table": "scale_shift_table_a2v_ca_video",
    "audio_a2v_cross_attn_scale_shift_table": "scale_shift_table_a2v_ca_audio",
    "norm_q": "q_norm",
    "norm_k": "k_norm",
}
LTX2_CONNECTORS_TO_ORIGINAL = {
    "video_connector": "video_embeddings_connector",
    "audio_connector": "audio_embeddings_connector",
    "transformer_blocks": "transformer_1d_blocks",
    "norm_q": "q_norm",
    "norm_k": "k_norm",
}


def write_ltx2_gguf(pipe, path):
    """
    Writes the transformer and text connectors of a diffusers LTX-2 pipeline like the files of unsloth/LTX-2.3-GGUF:
    original LTX-2 tensor names, without the text projections. Linear weights, scale/shift tables and registers are
    stored as BF16, which has to be dequantized, the other tensors as F32.
    """
    import gguf

    writer = gguf.GGUFWriter(str(path), arch="ltxv")
    components = ((pipe.transformer, LTX2_TRANSFORMER_TO_ORIGINAL), (pipe.connectors, LTX2_CONNECTORS_TO_ORIGINAL))
    for component, to_original in components:
        linear_weights = {f"{name}.weight" for name, m in component.named_modules() if isinstance(m, torch.nn.Linear)}
        for name, tensor in component.state_dict().items():
            if "text_proj_in" in name:
                continue
            as_bf16 = name in linear_weights or name.endswith(("scale_shift_table", "learnable_registers"))
            qtype = gguf.GGMLQuantizationType.BF16 if as_bf16 else gguf.GGMLQuantizationType.F32
            original_name = ".".join(to_original.get(part, part) for part in name.split("."))
            writer.add_tensor(original_name, gguf.quants.quantize(tensor.float().numpy(), qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.parametrize(
    ("model_id", "model_type"), [("optimum-intel-internal-testing/tiny-random-ltx2", "text-to-video")]
)
def test_video_model_hf_gguf(model_id, model_type, tmp_path):
    from diffusers import DiffusionPipeline

    gt_file = tmp_path / "gt.csv"
    model_path = tmp_path / "model"
    target_path = tmp_path / "target_model"
    gguf_path = tmp_path / "model.gguf"
    pipe = DiffusionPipeline.from_pretrained(model_id)
    # bf16 values, which the BF16 tensors of the GGUF file hold exactly
    for component in (pipe.transformer, pipe.connectors):
        for param in component.parameters():
            param.data = param.data.to(torch.bfloat16).float()
    pipe.save_pretrained(model_path)
    write_ltx2_gguf(pipe, gguf_path)
    # The target pipeline has other weights where the GGUF file has weights, so they must come from the file
    generator = torch.Generator().manual_seed(0)
    for component in (pipe.transformer, pipe.connectors):
        for name, param in component.named_parameters():
            if "text_proj_in" not in name:
                param.data = torch.randn(param.shape, generator=generator) * 0.02
    pipe.save_pretrained(target_path)

    run_wwb(
        [
            "--base-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--model-type",
            model_type,
            "--hf",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    output = run_wwb(
        [
            "--target-model",
            target_path,
            "--gguf-file",
            gguf_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--model-type",
            model_type,
            "--hf",
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
        ]
    )

    assert f"Loading the transformer and text connector weights from {gguf_path}" in output
    # the GGUF file holds the transformer and connector weights of the reference model exactly
    assert get_similarity(output) >= 0.999
