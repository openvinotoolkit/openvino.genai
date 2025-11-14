import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

RETRY_WAIT_SECONDS = 5


class AtomicDownloadManager:
    def __init__(self, final_path: Path, marker_name: str = ".download_complete"):
        self.final_path = Path(final_path)
        self.completion_marker = self.final_path / marker_name
        random_suffix = uuid.uuid4().hex[:8]
        self.temp_path = self.final_path.parent / f".tmp_{self.final_path.name}_{random_suffix}"

    def is_complete(self) -> bool:
        return self.completion_marker.exists()

    def execute(self, download_fn: Callable[[Path], None]) -> None:
        if self.is_complete():
            logger.info(f"Already complete (marker found): {self.final_path}")
            return
        
        if self.final_path.exists():
            logger.info(f"Removing existing directory before operation: {self.final_path}")
            self._remove_directory_with_retry(self.final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        try:
            download_fn(self.temp_path)
            self._move_to_final_location()
            self._mark_complete()
        except Exception:
            logger.exception("Error during operation")
            self._cleanup_temp()
            raise

    def _remove_directory_with_retry(self, path: Path, max_retries: int = 5) -> None:
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Failed to remove directory "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                        f"Retrying in {RETRY_WAIT_SECONDS}s..."
                    )
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    logger.exception(
                        f"Failed to remove directory after {max_retries}"
                    )
                    raise

    def _move_to_final_location(self) -> None:
        if self.final_path.exists():
            logger.info(
                "Destination already exists (created by another process), "
                f"skipping move: {self.final_path}"
            )
            return

        logger.info(f"Moving from temp to final location: {self.temp_path} -> {self.final_path}")
        
        try:
            shutil.move(str(self.temp_path), str(self.final_path))
        except Exception:
            logger.exception(f"Error during move - possibly due to concurrent access")

    def _mark_complete(self) -> None:
        self.completion_marker.touch()
        logger.info(f"Operation complete, marker created: {self.completion_marker}")

    def _cleanup_temp(self) -> None:
        if self.temp_path.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_path}")
            try:
                self._remove_directory_with_retry(self.temp_path, max_retries=3)
            except Exception:
                logger.exception("Could not clean up temp directory")
