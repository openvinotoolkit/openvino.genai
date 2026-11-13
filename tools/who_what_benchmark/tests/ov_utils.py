# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import uuid
import shutil
import subprocess  # nosec B404

from pathlib import Path
from importlib import metadata

sys.path.append(str(Path(__file__).resolve().parents[3] / "tests" / "python_tests"))

from utils.atomic_download import AtomicDownloadManager  # noqa: E402
from utils.constants import get_ov_cache_dir  # noqa: E402, F401
from utils.network import retry_request  # noqa: E402


def get_hf_cli_command() -> str:
    # huggingface_hub < 1.0 supports huggingface-cli, huggingface_hub >= 1.0 use hf
    version = metadata.version("huggingface_hub")
    major = int(version.split(".", 1)[0])
    if major >= 1:
        return "hf"
    return "huggingface-cli"


def download_hf_files_to_cache(repo_id: str, cache_dir: Path, filenames: list[str]) -> Path:
    """Download a set of files from a Hugging Face repo into a local cache directory.

    This helper is designed for tests that share a cache across CI jobs. If the
    destination directory already exists, it verifies that all required files are
    present and only downloads missing ones.

    Args:
        repo_id: Hugging Face repo id (e.g. "org/model").
        cache_dir: Destination directory.
        filenames: List of repo file paths to download.

    Returns:
        Path to the destination directory containing the downloaded files.
    """

    dest_dir = Path(cache_dir)

    def download_to_local_dir(local_dir: Path) -> None:
        for filename in filenames:
            command = [
                get_hf_cli_command(),
                "download",
                repo_id,
                filename,
                "--local-dir",
                str(local_dir),
            ]

            def _run_download() -> None:
                subprocess.run(command, check=True, text=True, encoding="utf-8", capture_output=True)

            retry_request(_run_download)

    # If destination exists (e.g. shared CI cache), make sure all required files are present.
    if dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        missing = [name for name in filenames if not (dest_dir / name).exists()]
        if missing:
            temp_dir = dest_dir.parent / f".tmp_{dest_dir.name}_{uuid.uuid4().hex[:8]}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                download_to_local_dir(temp_dir)
                for filename in missing:
                    src = temp_dir / filename
                    if not src.exists():
                        raise AssertionError(f"Download failed: {src}")
                    dst = dest_dir / filename
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(dst)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        manager = AtomicDownloadManager(dest_dir)
        manager.execute(download_to_local_dir)

    for filename in filenames:
        downloaded = dest_dir / filename
        if not downloaded.exists():
            raise AssertionError(f"Download failed: {downloaded}")

    return dest_dir
