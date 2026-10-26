import errno
from pathlib import Path

import pytest

from utils.atomic_download import AtomicDownloadManager


def test_failed_cache_publication_is_not_reported_as_success(tmp_path, monkeypatch):
    manager = AtomicDownloadManager(tmp_path / "model")
    attempts = []

    def download(destination):
        (destination / "config.json").write_text("{}", encoding="utf-8")

    def fail_rename(_source, _destination):
        attempts.append(True)
        raise OSError(errno.EIO, "Cache filesystem I/O failure")

    monkeypatch.setattr(Path, "rename", fail_rename)

    with pytest.raises(OSError, match="Cache filesystem I/O failure"):
        manager.execute(download)

    assert not manager.final_path.exists()
    assert not manager.temp_path.exists()
    assert len(attempts) == 3


def test_cache_is_published_and_reused(tmp_path):
    manager = AtomicDownloadManager(tmp_path / "model")
    calls = []

    def download(destination):
        calls.append(destination)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "weights.bin").write_bytes(b"weights")

    manager.execute(download)
    manager.execute(download)

    assert len(calls) == 1
    assert manager.is_complete()
    assert (manager.final_path / "config.json").read_text(encoding="utf-8") == "{}"
    assert (manager.final_path / "weights.bin").read_bytes() == b"weights"
    assert not manager.temp_path.exists()


@pytest.mark.parametrize("error_number", [errno.EIO, errno.EINTR, errno.ESTALE])
@pytest.mark.parametrize("failures", [1, 2])
def test_transient_publication_error_is_retried(tmp_path, monkeypatch, error_number, failures):
    manager = AtomicDownloadManager(tmp_path / "model")
    rename = Path.rename
    attempts = []

    def download(destination):
        (destination / "config.json").write_text("{}", encoding="utf-8")

    def flaky_rename(source, destination):
        attempts.append(True)
        if len(attempts) <= failures:
            raise OSError(error_number, "Transient cache filesystem failure")
        return rename(source, destination)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    manager.execute(download)

    assert len(attempts) == failures + 1
    assert (manager.final_path / "config.json").exists()
    assert not manager.temp_path.exists()


def test_concurrent_publisher_result_is_reused(tmp_path, monkeypatch):
    manager = AtomicDownloadManager(tmp_path / "model")

    def download(destination):
        (destination / "config.json").write_text("local", encoding="utf-8")

    def another_process_published(_source, destination):
        destination.mkdir()
        (destination / "config.json").write_text("other", encoding="utf-8")
        raise FileExistsError(errno.EEXIST, "Another process published the cache")

    monkeypatch.setattr(Path, "rename", another_process_published)
    manager.execute(download)

    assert (manager.final_path / "config.json").read_text(encoding="utf-8") == "other"
    assert not manager.temp_path.exists()


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.ENOSPC, errno.EXDEV])
def test_nontransient_publication_error_is_not_hidden(tmp_path, monkeypatch, error_number):
    manager = AtomicDownloadManager(tmp_path / "model")
    attempts = []

    def download(destination):
        (destination / "config.json").write_text("{}", encoding="utf-8")

    def fail_rename(_source, _destination):
        attempts.append(True)
        raise OSError(error_number, "Nontransient cache filesystem failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError) as caught:
        manager.execute(download)

    assert caught.value.errno == error_number
    assert len(attempts) == 1
    assert not manager.final_path.exists()
    assert not manager.temp_path.exists()


def test_download_error_is_propagated_and_temporary_files_are_removed(tmp_path):
    manager = AtomicDownloadManager(tmp_path / "model")

    def download(destination):
        (destination / "config.json").write_text("partial", encoding="utf-8")
        raise RuntimeError("Conversion failed")

    with pytest.raises(RuntimeError, match="Conversion failed"):
        manager.execute(download)

    assert not manager.final_path.exists()
    assert not manager.temp_path.exists()
