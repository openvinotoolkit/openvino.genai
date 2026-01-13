import logging
import shutil
import uuid
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class AtomicDownloadManager:
    def __init__(self, final_path: Path):
        self.final_path = Path(final_path)
        random_suffix = uuid.uuid4().hex[:8]
        self.temp_path = self.final_path.parent / f".tmp_{self.final_path.name}_{random_suffix}"

    def is_complete(self) -> bool:
        return self.final_path.exists()

    def execute(self, download_fn: Callable[[Path], None]) -> None:
        if self.is_complete():
            logger.info(f"Already downloaded: {self.final_path}")
            return

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        try:
            download_fn(self.temp_path)
            self._move_to_final_location()
        except Exception:
            logger.exception("Error during operation")
            self._cleanup_temp()
            raise

    def _move_to_final_location(self) -> None:
        if self.final_path.exists():
            logger.info(f"Destination already exists (created by another process): {self.final_path}")
            self._cleanup_temp()
            return

        logger.info(f"Moving temp to final location: {self.temp_path} -> {self.final_path}")
        try:
            self.temp_path.rename(self.final_path)
        except Exception:
            logger.warning(f"Rename failed, falling back to shutil.move")
            if self.final_path.exists():
                logger.info(f"Destination created by another process during rename attempt: {self.final_path}")
                self._cleanup_temp()
                return
            try:
                shutil.move(str(self.temp_path), str(self.final_path))
            except Exception:
                logger.exception("Error during move - assuming it was created successfully by another process")
                self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        if self.temp_path.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_path}")
            try:
                shutil.rmtree(self.temp_path)
            except Exception:
                logger.exception("Could not clean up temp directory")
