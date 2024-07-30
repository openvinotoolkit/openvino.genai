from pathlib import Path

import nncf

COMPRESSION_OPTIONS = {
    "INT8": {
        "mode": nncf.CompressWeightsMode.INT8 if "INT8_ASYM" not in nncf.CompressWeightsMode.__members__ else nncf.CompressWeightsMode.INT8_ASYM},
    "INT4_SYM": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
    },
    "INT4_ASYM": {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
    },
    "4BIT_MAXIMUM": {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
        "ratio": 1,
        "all_layers": True,
    },
    "E2M1": {
        "mode": nncf.CompressWeightsMode.E2M1,
        "group_size": 32,
        "all_layers": True,
    },
}

if "INT8_ASYM" in nncf.CompressWeightsMode.__members__:
    COMPRESSION_OPTIONS["INT8_ASYM"] = {"mode": nncf.CompressWeightsMode.INT8_ASYM}

if "INT8_SYM" in nncf.CompressWeightsMode.__members__:
    COMPRESSION_OPTIONS["INT8_SYM"] = {"mode": nncf.CompressWeightsMode.INT8_SYM}


def get_compressed_path(output_dir: str, base_precision, option: str):
    return Path(output_dir) / "pytorch/dldt/compressed_weights" / f"OV_{base_precision}-{option}"
