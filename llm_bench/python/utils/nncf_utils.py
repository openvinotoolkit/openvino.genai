from pathlib import Path

import nncf


COMPRESSION_OPTIONS = {
    "INT8": {"mode": nncf.CompressWeightsMode.INT8},
    "INT4_SYM": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
    },
    "INT4_ASYM": {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
    },
}


def get_compressed_path(output_dir: str, base_precision, option: str):
    return Path(output_dir) / "pytorch/dldt/compressed_weights" / f"OV_{base_precision}-{option}"


INT4_MODEL_CONFIGURATION = {
    "dolly-v2-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 32, "ratio": 0.5},
    "opt-6.7b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "red-pajama-incite-7b-instruct": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128},
    "zephyr-7b-beta": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.6},
    "llama-2-7b-chat": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128, "ratio": 0.8},
    "llama-2-13b-chat": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8},
    "stablelm-3b-4e1t": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128, "ratio": 0.8},
    "stablelm-epoch-3b-preview": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128, "ratio": 0.8},
    "chatglm-2b": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 128, "ratio": 0.72},
}
