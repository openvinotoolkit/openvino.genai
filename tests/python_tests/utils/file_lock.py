# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import errno
from pathlib import Path
from contextlib import contextmanager

class FileLock:
    def __init__(self, lock_file: str, timeout: float = 300):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self._lock_file_fd = None
        self._stale_lock_timeout = 3600
    
    def _is_lock_stale(self) -> bool:
        try:
            if not self.lock_file.exists():
                return False
            
            mtime = self.lock_file.stat().st_mtime
            age = time.time() - mtime
            return age > self._stale_lock_timeout
        except (OSError, FileNotFoundError):
            return False
    
    def acquire(self, poll_interval: float = 0.1):
        start_time = time.time()
        
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        while True:
            try:
                self._lock_file_fd = os.open(
                    str(self.lock_file),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                
                if self._is_lock_stale():
                    try:
                        self.lock_file.unlink()
                        continue
                    except FileNotFoundError:
                        continue
                
                if self.timeout is not None and (time.time() - start_time) >= self.timeout:
                    raise TimeoutError(f"Timeout acquiring lock on {self.lock_file}")
                
                time.sleep(poll_interval)
    
    def release(self):
        if self._lock_file_fd is not None:
            os.close(self._lock_file_fd)
            self._lock_file_fd = None
        
        try:
            self.lock_file.unlink()
        except FileNotFoundError:
            pass
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
    
    def __del__(self):
        self.release()

@contextmanager
def file_lock(lock_file: str, timeout: float = 300):
    lock = FileLock(lock_file, timeout)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()

