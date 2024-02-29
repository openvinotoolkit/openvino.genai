from pathlib import Path

import nncf


COMPRESSION_OPTIONS = {
    "INT8": {"mode": nncf.CompressWeightsMode.INT8 if "INT8_ASYM" not in nncf.CompressWeightsMode.__members__ else nncf.CompressWeightsMode.INT8_ASYM},
    "INT4_SYM": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
    },
    "INT4_ASYM": {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
    },
}


if "INT8_ASYM" in nncf.CompressWeightsMode.__members__:
    COMPRESSION_OPTIONS["INT8_ASYM"] = {"mode": nncf.CompressWeightsMode.INT8_ASYM}

if "INT8_SYM" in nncf.CompressWeightsMode.__members__:
    COMPRESSION_OPTIONS["INT8_SYM"] = {"mode": nncf.CompressWeightsMode.INT8_SYM}


def get_compressed_path(output_dir: str, base_precision, option: str):
    return Path(output_dir) / "pytorch/dldt/compressed_weights" / f"OV_{base_precision}-{option}"


INT4_MODEL_CONFIGURATION = {
    "dolly-v2-3b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 32, "ratio": 0.5},
    "gpt-j-6b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64},
    "opt-6.7b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8},
    "bloomz-7b1": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 32, "ratio": 0.6},
    "red-pajama-incite-7b-instruct": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128},
    "zephyr-7b-beta": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.6},
    "llama-2-7b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.6},
    "llama-2-7b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
    "llama-2-13b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stablelm-3b-4e1t": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stablelm-epoch-3b-preview": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "stable-zephyr-3b-dpo": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8},
    "stable-code-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "rocket-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
    "chatglm2-6b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.72},
    "qwen-7b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.6},
    "open-llama-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True},
    "falcon-7b-instruct": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True},
    "orca-mini-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True},
    "mixtral-8x7b-v0.1": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
}
