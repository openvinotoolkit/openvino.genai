from pathlib import Path
import os
import tempfile
from datetime import datetime
from importlib import metadata


SHOULD_CLEANUP = os.environ.get("CLEANUP_CACHE", "").lower() in ("1", "true", "yes")


def get_wwb_cache_dir(temp_dir=tempfile.TemporaryDirectory()) -> Path:
    if "OV_CACHE" in os.environ:
        date_subfolder = datetime.now().strftime("%Y%m%d")
        ov_cache = os.path.join(os.environ["OV_CACHE"], date_subfolder)
        try:
            optimum_intel_version = metadata.version("optimum-intel")
            transformers_version = metadata.version("transformers")
            ov_cache = os.path.join(ov_cache, f"optimum-intel-{optimum_intel_version}_transformers-{transformers_version}")
        except metadata.PackageNotFoundError:
            pass
    else:
        ov_cache = temp_dir.name
    return Path(ov_cache).joinpath("wwb_cache")


WWB_CACHE_PATH = get_wwb_cache_dir()
