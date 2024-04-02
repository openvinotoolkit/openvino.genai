from pathlib import Path
import logging as log

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


def get_int4_default_compression_args(model_id):
    if model_id in INT4_MODEL_CONFIGURATION:
        compression_args = INT4_MODEL_CONFIGURATION[model_id]
    else:
        compression_args = COMPRESSION_OPTIONS["INT4_SYM"]
        log.info(f"Model is not supported with 4BIT_DEFAULT. Compress weights configuration was switched to INT4_SYM.")
    return compression_args


def get_compressed_path(output_dir: str, base_precision: str, option: str):
    output_dir = Path(output_dir)
    if option == "4BIT_DEFAULT":
        model_id = Path(output_dir).parents[3].name
        option = get_int4_default_compression_args(model_id)["mode"].split(".")[-1].upper()
    return output_dir / "pytorch" / "dldt" / "compressed_weights" / f"OV_{base_precision}-{option}"


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
    "stablelm-3b-4e1t": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8,
                         "dataset": {"name": "wikitext,wikitext-2-v1,train[:1000],text", "awq": True}},
    "stablelm-epoch-3b-preview": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8,
                                  "dataset": {"name": "wikitext,wikitext-2-v1,train[:1000],text", "awq": True}},
    "stable-zephyr-3b-dpo": {"mode": nncf.CompressWeightsMode.INT4_ASYM, "group_size": 64, "ratio": 0.8,
                             "dataset": {"name": "wikitext,wikitext-2-v1,train[:1000],text", "awq": True}},
    "stable-code-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8},
    "rocket-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
    "chatglm2-6b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.72},
    "qwen-7b-chat": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.6},
    "open-llama-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True},
    "falcon-7b-instruct": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True},
    "orca-mini-3b": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "all_layers": True,
                     "dataset": {"name": "wikitext,wikitext-2-v1,train[:1000],text", "awq": False}},
    "bloomz-560m": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 64, "ratio": 0.8,
                    "dataset": {"name": "wikitext,wikitext-2-v1,train[:1000],text", "awq": True}},
    "mixtral-8x7b-v0.1": {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": 128, "ratio": 0.8},
}
