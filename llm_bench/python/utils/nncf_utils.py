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
