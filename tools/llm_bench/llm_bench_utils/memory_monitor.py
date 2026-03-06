# Copyright (C) 2025-2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import atexit
import queue
import threading
import multiprocessing
from multiprocessing import Process as mProcess
from psutil import Process as psutilProcess
import time
import os
from enum import Enum
from functools import lru_cache
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from collections import OrderedDict

import math
import psutil
import matplotlib
import matplotlib.pyplot as plt
import logging as log
import traceback
import json
import sys


# CUSTOM FIX TO AVOID ISSUE: RuntimeError: main thread is not in main loop
matplotlib.use("Agg")

class MonitorLevel(Enum):
    DISABLED = 0
    WARMUP = 1
    FULL = 2

class MonitorType(Enum):
    DISABLED = 0
    THREAD = 1
    PROCESS = 2

class MonitorMode(Enum):
    NO_MONITORING = (MonitorLevel.DISABLED, MonitorType.DISABLED)
    THREAD_WARMUP = (MonitorLevel.WARMUP, MonitorType.THREAD)
    THREAD_FULL = (MonitorLevel.FULL, MonitorType.THREAD)
    PROCESS_WARMUP = (MonitorLevel.WARMUP, MonitorType.PROCESS)
    PROCESS_FULL = (MonitorLevel.FULL, MonitorType.PROCESS)

    def __init__(self, monitor_level, monitor_type):
        self.monitor_level = monitor_level
        self.monitor_type = monitor_type

    @classmethod
    def from_code(cls, code: int):
        """Create MonitorMode from integer code 0-4"""
        modes = [
            cls.NO_MONITORING,
            cls.THREAD_WARMUP,
            cls.THREAD_FULL,
            cls.PROCESS_WARMUP,
            cls.PROCESS_FULL,
        ]
        if not 0 <= code <= 4:
            raise ValueError(f"Invalid memory monitor mode: {code}. Must be 0-4")
        return modes[code]

    @property
    def is_process(self) -> bool:
        return self.monitor_type == MonitorType.PROCESS

    @property
    def is_thread(self) -> bool:
        return self.monitor_type == MonitorType.THREAD

    @property
    def is_full(self) -> bool:
        return self.monitor_level == MonitorLevel.FULL

    @property
    def is_warmup(self) -> bool:
        return self.monitor_level == MonitorLevel.WARMUP

    @property
    def is_enabled(self) -> bool:
        return self.monitor_level != MonitorLevel.DISABLED


class MemoryMonitorHandler:
    def __init__(self, args):
        self.mode = MonitorMode.from_code(args.memory_consumption)
        self.mth = MemThreadHandler(args) if self.mode.is_thread else None
        self.mmh = MemoryMarkerHandler(args) if self.mode.is_process else None
        self.last_iter_number = None
        self.args = args

        self.log_data = self._noop_log_data
        if self.mth:
            self.log_data = self.mth.log_data
        self.update_marker = self._noop_update_marker
        if self.mmh:
            self.update_marker = self.mmh.update_marker
        self.stop_and_collect_data = self._noop_stop_and_collect_data
        if self.mth:
            self.stop_and_collect_data = self.mth.stop_and_collect_data
        self.get_initial_mem_data = self._noop_get_mem_data
        if self.mth:
            self.get_initial_mem_data = self.mth.get_initial_mem_data
        self.get_compilation_mem_data = self._noop_get_mem_data
        if self.mth:
            self.get_compilation_mem_data = self.mth.get_compilation_mem_data

    def start(self, iter_num=None):
        if self.mth:
            if self.mode.is_full or iter_num is None:
                return self.mth.start()
            elif self.mode.is_warmup and iter_num == 0:
                return self.mth.start()
        if self.last_iter_number is not None and iter_num is not None:
            if self.mode.is_warmup and self.last_iter_number != iter_num:
                return self.mmh.stop()

    @staticmethod
    def _noop_update_marker(marker):
        """No-op marker update when process monitoring is disabled"""
        pass

    @staticmethod
    def _noop_stop_and_collect_data(stage):
        """No-op stop when process monitoring is disabled"""
        pass

    @staticmethod
    def _noop_get_mem_data(print_unit=None):
        """No-op get data  when process monitoring is disabled"""
        return {}

    @staticmethod
    def _noop_log_data(compilation=False):
        """No-op stop when process monitoring is disabled"""
        pass

    def stop(self):
        if self.mmh:
            return self.mmh.stop()
        if self.mth:
            return self.mth.stop()

    def iter_stop_and_collect_data(self, iter_num, dict_format=True):
        cooldown = self.args.memory_consumption_cooldown
        if cooldown is not None:
            self.mmh.update_marker("cooldown")
            time.sleep(cooldown)
        self.last_iter_number = iter_num
        if self.mode.is_process or not self.mode.is_enabled:
            return {} if dict_format else [""] * 4
        if self.mode.is_warmup and iter_num > 0:
            return {} if dict_format else [""] * 4

        dir_name = "warm-up"
        if iter_num > 0:
            dir_name = f"P{iter_num}"
        self.stop_and_collect_data(dir_name)
        return self.mth.get_data(dict_format)


######################################################
### Memory Monitoring (in separated thread)

class MemoryType(Enum):
    RSS = "rss"
    SYSTEM = "system"


class MemStatus:
    def __init__(self, rss=None, sys=None):
        self.rss = rss
        self.sys = sys


class MemoryUnit(Enum):
    B = "B"  # byte
    KiB = "KiB"  # Kibibyte
    MiB = "MiB"  # Mibibyte
    GiB = "GiB"  # Gibibyte
    KB = "KB"  # Kilobyte
    MB = "MB"  # Megabyte
    GB = "GB"  # Gigabyte


@lru_cache
def system_memory_warning():
    # Log once
    log.warning(
        "Please note that MemoryType.SYSTEM in general is affected by other processes that change RAM availability."
    )


class MemoryMonitor:
    def __init__(
        self,
        interval: Optional[float] = 0.1,
        memory_type: Optional[MemoryType] = MemoryType.RSS,
        memory_unit: Optional[MemoryUnit] = MemoryUnit.MiB,
        include_child_processes: Optional[bool] = None,
    ):
        """
        Memory monitoring utility to measure python process memory footprint. After start() is called, it
        creates a thread which runs in parallel and takes memory measurements every *interval* seconds using the
        specified *memory_type* approach. When stop() is called, the memory measuring thread is stopped. The results
        can be obtained by calling get_data(). Memory logs can be saved by calling save_memory_logs(). There are two
        log files: one with data values in a .txt format and another one in a form of a 2D time-memory plot.

        Memory monitor itself allocates some memory itself, especially during figure saving. It is advised to use it
        for measuring large memory processes.

        :param interval: How frequently to take memory measurements (in seconds).
        :param memory_type: Type of memory to log. Accepts four possible values:
            - MemoryType.RSS: Resident Set Size is the portion of memory occupied by a process that is held in RAM.
              Values are obtained through psutil library. If some data is read using mmap, RSS will report this data
              as allocated, however this is not necessarily the case.
            - MemoryType.SYSTEM: This metric is defined as the difference between total system virtual memory
              and system available memory. Be aware, that this way it is affected by other processes that can change
              RAM availability. It is advised to call get_data(memory_from_zero=True) for this type of memory logging,
              if one is interested in memory footprint for a certain process. This subtracts the starting memory from
              all values.

            RSS and SYSTEM behave differently when mmap is used, e.g. during OV model loading. RSS will report data
            which was read with mmap enabled as allocated, however this is not necessarily the case. SYSTEM does not
            report memory loaded with mmap. So it can be used to analyze "pure" memory usage without contribution of
            mmap pages which are actually free, but are reported as allocated by RSS.
        :param memory_unit: Unit to report memory in.
        :param include_child_processes: For MemoryType.RSS only: whether to include memory of child processes. If not
            provided, child processes are counted.
        """
        self.interval = interval
        self.timestamp = int(time.time())
        self.memory_type = memory_type
        if memory_type == MemoryType.SYSTEM:
            system_memory_warning()
        elif memory_type == MemoryType.RSS:
            if include_child_processes is None:
                include_child_processes = True
        else:
            raise ValueError("Unknown memory type to log")
        self.memory_unit = memory_unit
        self.include_child_processes = include_child_processes

        self._monitoring_thread_should_stop = False
        self._monitoring_in_progress = False

        self._memory_monitor_thread = None
        self._memory_values_queue = None
        self._stop_logging_atexit_fn = None

    def start(self, at_exit_fn: Optional[Callable] = None) -> "MemoryMonitor":
        """
        Start memory monitoring.

        :param at_exit_fn: A callable to execute at program exit. Useful for providing logs saving routine, e.g.
            ```
                at_exit_fn = lambda: memory_monitor.save_memory_logs(*memory_monitor.get_data(), save_dir)
                memory_monitor.start(at_exit_fn=at_exit_fn)
            ```
        """
        if self._monitoring_in_progress:
            log.warning(
                f"Monitoring was already in progress. MemoryMonitor will be restarted and previous data will be lost for {self.memory_type}."
            )
            self.stop()

        self._memory_values_queue = queue.Queue()
        self._monitoring_thread_should_stop = False

        self._memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self._memory_monitor_thread.daemon = True
        self._memory_monitor_thread.start()
        if at_exit_fn:
            self._stop_logging_atexit_fn = at_exit_fn
            atexit.register(self._stop_logging_atexit_fn)

        self._monitoring_in_progress = True

        return self

    def stop(self):
        """
        Stop memory monitoring.
        """
        if not self._monitoring_in_progress:
            return
        self._monitoring_thread_should_stop = True
        self._monitoring_in_progress = False
        self._memory_monitor_thread.join()
        if self._stop_logging_atexit_fn is not None:
            atexit.unregister(self._stop_logging_atexit_fn)
            self._stop_logging_atexit_fn = None

    def get_data(self, memory_from_zero: Optional[bool] = False) -> tuple[list, list]:
        """
        :param memory_from_zero: Whether to normalize memory measurements by subtracting the first value. This way
            the measurements will start with 0. Hence, is not very reliable and may actually result in negative values.
        :returns: A tuple of list where the first element corresponds to measurements timestamps and the second one --
        to memory values.
        """
        memory_usage_data = list(self._memory_values_queue.queue)
        if len(memory_usage_data) == 0:
            return [], []
        time_values, memory_values = tuple(zip(*memory_usage_data))
        time_values = _subtract_first_element(list(time_values))
        if memory_from_zero:
            memory_values = _subtract_first_element(list(memory_values))

        # Convert to target memory unit
        memory_values = list(map(partial(cast_bytes_to, memory_unit=self.memory_unit), memory_values))

        return time_values, memory_values

    def save_memory_logs(
        self,
        time_values: list[float],
        memory_values: list[float],
        save_dir: Path,
        plot_title: Optional[str] = "",
        filename_suffix: Optional[str] = "",
    ):
        """
        Save memory logs as a text file and a 2D plot.

        :param time_values: Timestamps of the memory measurements.
        :param memory_values: Memory measurements.
        :param save_dir: Directory to save logs into.
        :param plot_title: A title for a plot.
        :param filename_suffix: A string suffix to give to the saved files.
        """
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        filename_label = f"{self.memory_type.value}_memory_usage{filename_suffix}"
        # Save measurements to text file
        counter = 0
        while True:
            log_filepath = save_dir / f"{filename_label}_{counter}.txt"
            if os.path.exists(log_filepath):
                counter += 1
            else:
                break

        with open(log_filepath, "w") as log_file:
            if len(time_values) == 0:
                log_file.write("No measurements recorded.\nPlease make sure logging duration or interval were enough.")
                return
            for timestamp, memory_usage in zip(time_values, memory_values):
                log_file.write(f"{timestamp} {memory_usage:.3f}\n")

            log_file.writelines(
                [
                    f"Timestamp: {self.timestamp}\n",
                    f"Total time: {time_values[-1] - time_values[0]}\n",
                    f"Max memory: {max(memory_values):.3f} ({self.memory_unit.value})",
                ]
            )

        # Save measurements plot
        self.save_memory_plot(log_filepath, plot_title)

    def save_memory_plot(self, log_filepath: Path, plot_title: Optional[str] = "", filename_suffix: Optional[str] = ""):
        """
        Parse pre-saved txt file logs and plot a new figure based on this data. May be useful for re-plotting with
        different title.

        :param log_filepath: A path to a .txt log file.
        :param plot_title: A title to give to a plot.
        :param filename_suffix: A string suffix to give to the saved figure.
        """
        with open(log_filepath, "r") as f:
            lines = f.readlines()
            time_values, memory_values = [], []
            for line in lines[:-3]:
                time_value, memory_value = tuple(map(float, line.split(" ")))
                time_values.append(time_value)
                memory_values.append(memory_value)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(time_values, memory_values)
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"Memory Usage ({self.memory_type.value}, {self.memory_unit.value})")
        plt.title(f"{plot_title} Max_{self.memory_type.value}: {max(memory_values):.2f} {self.memory_unit.value}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(log_filepath).replace(".txt", f"{filename_suffix}.png"))
        plt.close(fig)

    def __enter__(self) -> "MemoryMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def get_system_memory():
        return psutil.virtual_memory().total - psutil.virtual_memory().available

    @staticmethod
    def get_rss_memory(include_child_processes=True):
        this_process = psutilProcess()
        bytes_used = this_process.memory_info().rss
        if include_child_processes:
            return sum((proc.memory_info().rss for proc in this_process.children(recursive=True)), start=bytes_used)
        return bytes_used

    def _monitor_memory(self):
        self.timestamp = int(time.time())
        while not self._monitoring_thread_should_stop:
            _last_measurement_time = time.perf_counter()
            if self.memory_type == MemoryType.RSS:
                bytes_used = MemoryMonitor.get_rss_memory(self.include_child_processes)
            elif self.memory_type == MemoryType.SYSTEM:
                bytes_used = MemoryMonitor.get_system_memory()
            else:
                raise Exception("Unknown memory type to log")
            if self._monitoring_thread_should_stop:
                break
            self._memory_values_queue.put((time.perf_counter(), bytes_used))
            time.sleep(max(0.0, self.interval - (time.perf_counter() - _last_measurement_time)))


class MemThreadHandler:
    MEMORY_NOT_COLLECTED = ""
    DEF_MEM_UNIT = MemoryUnit.MiB

    def __init__(self, args):
        self.interval = args.memory_consumption_interval
        self.memory_unit = MemoryUnit.MiB
        self.proc_id = os.getpid()

        self.save_dir = None
        if args.memory_consumption_dir:
            self.set_dir(args.memory_consumption_dir)
        self.initial_mem_status = self.log_curent_memory_data(prefix="Start")
        self.compilation_mem_info = {
            "max_mem": MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED),
            "increase_mem": MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED),
        }

        self.memory_data = {"full_mem": {}, "from_zero": {}}
        self.memory_types = [MemoryType.RSS, MemoryType.SYSTEM]
        self.memory_monitors = {}
        self.memory_data = {"full_mem": {}, "from_zero": {}}

    def create_monitors(self):
        for memory_type in self.memory_types:
            self.memory_monitors[memory_type] = MemoryMonitor(
                interval=self.interval, memory_type=memory_type, memory_unit=self.memory_unit
            )

    def set_dir(self, dir):
        if not Path(dir).exists():
            log.warning(f"Path to dir for memory consumption data is not exists {dir}, run without it.")
        else:
            self.save_dir = Path(dir)

    def start(self, delay=None):
        self.memory_data = {"full_mem": {}, "from_zero": {}}
        for mm in self.memory_monitors.values():
            mm.start()

        # compilation could be very fast, apply delay
        if delay is not None:
            time.sleep(delay)
        else:
            time.sleep(self.interval * 3)

    def stop_and_collect_data(self, dir_name="mem_monitor_log"):
        dir_name = f"{dir_name}_{self.proc_id}"
        self.stop()

        for mt, mm in self.memory_monitors.items():
            if not mm._memory_values_queue or len(mm._memory_values_queue.queue) == 0:
                continue

            for from_zero in [False, True]:
                time_values, memory_values = mm.get_data(memory_from_zero=from_zero)

                mm_measure_type = "from_zero" if from_zero else "full_mem"
                self.memory_data[mm_measure_type][mt] = max(memory_values)

                if self.save_dir:
                    mm.save_memory_logs(
                        time_values,
                        memory_values,
                        save_dir=self.save_dir / dir_name,
                        filename_suffix="_mem_increase" if from_zero else "",
                    )

    def stop(self):
        # Stop addition of new values as soon as possible
        for mm in self.memory_monitors.values():
            mm._monitoring_thread_should_stop = True

        for mm in self.memory_monitors.values():
            mm.stop()

    def log_curent_memory_data(self, prefix : str = ""):
        mem_status = MemStatus(cast_bytes_to(MemoryMonitor.get_rss_memory(), self.memory_unit),
                               cast_bytes_to(MemoryMonitor.get_system_memory(), self.memory_unit))
        log.info(f"{prefix} RSS memory {mem_status.rss:.2f}{self.memory_unit.value}, "
                 f"{prefix} System memory {mem_status.sys:.2f}{self.memory_unit.value}")
        return mem_status

    def log_data(self, compilation: bool = False):
        max_rss_mem, rss_increase, max_sys_mem, sys_increase = self.get_data()
        comment = ""
        if compilation:
            comment = "on compilation phase"
            self.compilation_mem_info["max_mem"].sys = max_sys_mem
            self.compilation_mem_info["max_mem"].rss = max_rss_mem
            self.compilation_mem_info["increase_mem"].sys = sys_increase
            self.compilation_mem_info["increase_mem"].rss = rss_increase

        msg = (f"Max RSS memory {comment}: {max_rss_mem:.2f}{self.memory_unit.value}, "
               f"RSS memory increase {comment}: {rss_increase:.2f}{self.memory_unit.value}, "
               f"Max System memory {comment}: {max_sys_mem:.2f}{self.memory_unit.value}, "
               f"System memory increase {comment}: {sys_increase:.2f}{self.memory_unit.value}")
        log.info(msg)

    def get_data(self, dict_format=False):
        if dict_format:
            bytes_total = cast_bytes_to(psutil.virtual_memory().total, memory_unit=self.memory_unit)
            max_sys_mem_full = float(self.memory_data["full_mem"].get(MemoryType.SYSTEM, -1))
            max_rss_mem_full = float(self.memory_data["full_mem"].get(MemoryType.RSS, -1))
            return {
                "max_sys_mem": max_sys_mem_full,
                "max_sys_mem_share": 100.0 * max_sys_mem_full / bytes_total,
                "max_sys_mem_increase": self.memory_data["from_zero"].get(MemoryType.SYSTEM, -1),
                "max_rss_mem": max_rss_mem_full,
                "max_rss_mem_share": 100.0 * max_rss_mem_full / bytes_total,
                "max_rss_mem_increase": self.memory_data["from_zero"].get(MemoryType.RSS, -1),
            }
        return (
            self.memory_data["full_mem"].get(MemoryType.RSS, -1),
            self.memory_data["from_zero"].get(MemoryType.RSS, -1),
            self.memory_data["full_mem"].get(MemoryType.SYSTEM, -1),
            self.memory_data["from_zero"].get(MemoryType.SYSTEM, -1),
        )


class MemStatus:
    def __init__(self, rss=None, sys=None):
        self.rss = rss
        self.sys = sys


class MemorySampler5:
    chunk_size = 8192
    header = "rss", "uss", "priv", "sys", "nsys"

    def __init__(self, process_id, unit_value=1048576, unit_name="MiB"):
        self.process_id = process_id
        self.unit_val = int(unit_value)
        self.unit_name = unit_name

    def collect(self):
        parent_process = psutilProcess(self.process_id)
        mem_info = parent_process.memory_full_info()
        rss_mem, uss_mem, priv_mem = mem_info.rss, 0, 0
        if hasattr(mem_info, "uss"):
            uss_mem = mem_info.uss
        if hasattr(mem_info, "private"):
            priv_mem = mem_info.private
        for child_proc in parent_process.children(recursive=True):
            child_mem_info = child_proc.memory_full_info()
            rss_mem += child_mem_info.rss
            if hasattr(child_mem_info, "uss"):
                uss_mem += child_mem_info.uss
            if hasattr(child_mem_info, "private"):
                priv_mem += child_mem_info.private
        sys_mem = psutil.virtual_memory().total - psutil.virtual_memory().available
        nsys_mem = round(100 * float(sys_mem) / psutil.virtual_memory().total, 1)
        sys_mem = round(float(sys_mem) / self.unit_val, 3)
        rss_mem = round(float(rss_mem) / self.unit_val, 3)
        uss_mem = round(float(uss_mem) / self.unit_val, 3)
        priv_mem = round(float(priv_mem) / self.unit_val, 3)
        return rss_mem, uss_mem, priv_mem, sys_mem, nsys_mem


class MemoryMarkerMonitor(list):
    def __init__(self, conn, process_id, sampling_interval, path_prefix):
        self.sampler = MemorySampler5(process_id)
        self.sampling_interval = float(sampling_interval)
        self.process_id = int(process_id)
        self.last_ts = time.perf_counter()
        self.marker = "start"
        self.conn = conn

        self.path_prefix = Path(f"{path_prefix}_{process_id}")
        if not self.path_prefix.exists():
            self.path_prefix.mkdir(parents=True)
        self.file_counter = 0
        self.collect_samples()

    def deduce_filename(self):
        while Path(self.path_prefix / f"{self.file_counter}.txt").exists():
            self.file_counter += 1
        return self.path_prefix / f"{self.file_counter}.txt"

    def loop(self):
        while True:
            last_interval = time.perf_counter() - self.last_ts
            wait_interval = max(0.0, self.sampling_interval - last_interval)
            if self.conn.poll(wait_interval):
                self.marker = self.conn.recv()
                if self.marker == "stop":
                    print("Memory worker: Received stop signal")
                    self.collect_samples()
                    break
            if len(self) >= self.sampler.chunk_size:
                self.write_chunk()
            try:
                self.collect_samples()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                print("Memory worker: Received stop signal")
                break
        self.write_chunk()

    def write_chunk(self):
        if not self:
            return

        counter = 0
        fname = self.deduce_filename()
        with open(fname, "w", encoding="utf-8") as fd:
            fd.write(f"#ts marker {' '.join(self.sampler.header)}\n")
            while self:
                row = self.pop(0)
                line = " ".join(map(str, row))
                fd.write(f"{line}\n")
                counter += 1
            fd.write(f"#number of samples: {counter}")
            fd.write(f" (in {self.sampler.unit_name})\n")

    def collect_samples(self):
        vals = self.sampler.collect()
        ts = round(self.last_ts, 4)
        self.last_ts = time.perf_counter()
        self.append((ts, self.marker, *vals))


class MemoryMarkerHandler:
    def __init__(self, args):
        def mem_worker(conn, pid, interval, path):
            mmm = MemoryMarkerMonitor(conn, pid, interval, path)
            try:
                mmm.loop()
                conn.close()
                print("Background worker stopped")
            except Exception as e:
                print(f"Error in background worker: {e}")

        report_path = args.memory_consumption_dir
        s_interval = args.memory_consumption_interval

        parent_pid = os.getpid()
        self.parent_conn, child_conn = multiprocessing.Pipe()
        pargs = child_conn, parent_pid, s_interval, report_path
        self.background_process = mProcess(target=mem_worker, args=pargs)
        self.background_process.start()

    def update_marker(self, marker):
        print(">>>>>>>>>>>>>>>>>> SEND MARKER", marker)
        self.parent_conn.send(marker)

    def stop(self):
        self.parent_conn.send("stop")
        self.background_process.join()
        self.parent_conn.close()


class MemoryPhantom:
    def __init__(self):
        self.mmh = None

    def smart_start(self, num=None):
        pass

    def update_marker(self, marker):
        pass

    def smart_stop_and_collect_data(self, num, dict_format=True):
        return {} if dict_format else [""] * 4

    def stop(self):
        pass


class MemoryDataSummarizer:
    MEMORY_NOT_COLLECTED = ""
    DEF_MEM_UNIT = MemoryUnit.MiB

    def __init__(self, args):
        memory_monitor = MemMonitorWrapper()
        self.memory_consumption_mode = args.memory_consumption
        memory_monitor.interval = args.memory_consumption_interval
        self.cooldown = args.memory_consumption_cooldown
        memory_monitor.create_monitors()
        if args.memory_consumption_dir and self.memory_consumption_mode < 3:
            memory_monitor.set_dir(args.memory_consumption_dir)
        self.memory_monitor = memory_monitor
        self.iteration_mem_data = []
        self.compilation_mem_info = {
            "max_mem": MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED),
            "increase_mem": MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED),
        }
        self.initial_mem_status = self.log_curent_memory_data(prefix="Start")

        self.mmh = None
        if self.memory_consumption_mode in (3, 4):
            self.mmh = MemoryMarkerHandler(args)

    def update_marker(self, marker):
        if self.mmh is not None:
            self.mmh.update_marker(marker)

    def verify_num_in_mode(self, num):
        if self.memory_consumption_mode not in (1, 2):
            return False
        if self.memory_consumption_mode == 1 and num > 0:
            return False
        return True

    def start(self, num=None):
        if self.cooldown is not None:
            time.sleep(int(self.cooldown))
        self.memory_monitor.start()

    def smart_start(self, num=None):
        if num is not None:
            if not self.verify_num_in_mode(num):
                return
        self.start()

    def stop_and_collect_data(self, dir_name="mem_monitor_log"):
        self.memory_monitor.stop_and_collect_data(dir_name)

    def smart_stop_and_collect_data(self, num, dict_format=True):
        if not self.verify_num_in_mode(num):
            return {} if dict_format else []

        if isinstance(num, int):
            dir_name = f"{'P' + str(num) if num > 0 else 'warm-up'}"
            self.memory_monitor.stop_and_collect_data(dir_name)
            return self.get_data(dict_format)
        return {} if dict_format else []

    def stop(self):
        self.memory_monitor.stop()

    def get_data(self, dict_format=False):
        return self.memory_monitor.get_data(dict_format)

    def log_data(self, compilation_phase: bool = False):
        max_rss_mem, rss_increase, max_sys_mem, sys_increase = self.memory_monitor.get_data()
        comment = ""
        if compilation_phase:
            comment = "on compilation phase"
            self.compilation_mem_info["max_mem"].sys = max_sys_mem
            self.compilation_mem_info["max_mem"].rss = max_rss_mem
            self.compilation_mem_info["increase_mem"].sys = sys_increase
            self.compilation_mem_info["increase_mem"].rss = rss_increase

        msg = (
            f"Max RSS memory {comment}: {max_rss_mem:.2f}{self.memory_monitor.memory_unit.value}, "
            f"RSS memory increase {comment}: {rss_increase:.2f}{self.memory_monitor.memory_unit.value}, "
            f"Max System memory {comment}: {max_sys_mem:.2f}{self.memory_monitor.memory_unit.value}, "
            f"System memory increase {comment}: {sys_increase:.2f}{self.memory_monitor.memory_unit.value}"
        )
        log.info(msg)

    def log_curent_memory_data(self, prefix: str = ""):
        mem_status = MemStatus(
            cast_bytes_to(MemoryMonitor.get_rss_memory(), self.memory_monitor.memory_unit),
            cast_bytes_to(MemoryMonitor.get_system_memory(), self.memory_monitor.memory_unit),
        )
        log.info(
            f"{prefix} RSS memory {mem_status.rss:.2f}{self.memory_monitor.memory_unit.value}, "
            f"{prefix} System memory {mem_status.sys:.2f}{self.memory_monitor.memory_unit.value}"
        )
        return mem_status

    def get_initial_mem_data(self, print_unit: MemoryUnit | None = None):
        suffix = f"({print_unit})" if print_unit else ""
        sys = self.initial_mem_status.sys
        rss = self.initial_mem_status.rss
        if print_unit and print_unit != self.memory_unit:
            sys = convert_mem_unit(sys, self.memory_unit, print_unit)
            rss = convert_mem_unit(rss, self.memory_unit, print_unit)
        return {
            f"initial_sys_mem{suffix}": round(sys, 5),
            f"initial_rss_mem{suffix}": round(rss, 5),
        }

    def get_compilation_mem_data(self, print_unit: MemoryUnit | None = None):
        bytes_total = cast_bytes_to(psutil.virtual_memory().total, memory_unit=self.memory_unit)

        suffix = f"({print_unit.value})" if print_unit else ""
        sys_max = self.compilation_mem_info["max_mem"].sys
        rss_max = self.compilation_mem_info["max_mem"].rss
        sys_increase = self.compilation_mem_info["increase_mem"].sys
        rss_increase = self.compilation_mem_info["increase_mem"].rss
        sys_share = 100.0 * float(self.compilation_mem_info["max_mem"].sys) / bytes_total
        rss_share = 100.0 * float(self.compilation_mem_info["max_mem"].rss) / bytes_total

        if print_unit and print_unit != self.memory_unit:
            sys_max = convert_mem_unit(sys_max, self.memory_unit, print_unit)
            rss_max = convert_mem_unit(rss_max, self.memory_unit, print_unit)
            sys_increase = convert_mem_unit(sys_increase, self.memory_unit, print_unit)
            rss_increase = convert_mem_unit(rss_increase, self.memory_unit, print_unit)

        return {
            f"compile_max_rss_mem{suffix}": round(rss_max, 5),
            f"compile_max_sys_mem{suffix}": round(sys_max, 5),
            f"compile_max_increase_rss_mem{suffix}": round(rss_increase, 5),
            f"compile_max_increase_sys_mem{suffix}": round(sys_increase, 5),
            "compile_max_share_rss_mem": round(rss_share, 3),
            "compile_max_share_sys_mem": round(sys_share, 3),
        }


def cast_bytes_to(bytes, memory_unit, round_to_int=False):
    memory_unit_divisors = {
        MemoryUnit.B: 1,
        MemoryUnit.KiB: 2**10,
        MemoryUnit.MiB: 2**20,
        MemoryUnit.GiB: 2**30,
        MemoryUnit.KB: 10**3,
        MemoryUnit.MB: 10**6,
        MemoryUnit.GB: 10**9,
    }
    result = bytes / memory_unit_divisors[memory_unit]
    return int(result) if round_to_int else result


def convert_mem_unit(amount, current_memory_unit, new_memory_unit, round_to_int=False):
    memory_unit_divisors = {
        MemoryUnit.B: 1,
        MemoryUnit.KiB: 2**10,
        MemoryUnit.MiB: 2**20,
        MemoryUnit.GiB: 2**30,
        MemoryUnit.KB: 10**3,
        MemoryUnit.MB: 10**6,
        MemoryUnit.GB: 10**9,
    }
    bytes = amount * memory_unit_divisors[current_memory_unit]
    result = bytes / memory_unit_divisors[new_memory_unit]
    return int(result) if round_to_int else result


def _subtract_first_element(data):
    for i in range(1, len(data)):
        data[i] = data[i] - data[0]
    data[0] = 0
    return data


######################################################
### Memory Marker Monitoring (in separated process)

class MemorySampler5(dict):
    chunk_size = 8192
    metrics = OrderedDict([
        ("rss", {"denom": 1048576, "unit": "MiB", "digits": 3, "cv": True}),
        ("uss", {"denom": 1048576, "unit": "MiB", "digits": 3, "cv": True}),
        ("priv", {"denom": 1048576, "unit": "MiB", "digits": 3, "cv": True}),
        ("sys", {"denom": 1048576, "unit": "MiB", "digits": 3, "cv": True}),
        ("nsys", {"denom": 1, "unit": "%", "digits": 1, "cv": False}),
    ])

    def __init__(self, process_id):
        dict.__init__(self)
        self.process_id = process_id
        self.header = self.make_header()

    def make_header(self):
        tmplist = []
        for k, v in self.metrics.items():
            tmplist.append(f"{k}({v['unit']})")
        return " ".join(tmplist)

    def init_marker(self, marker):
        self[marker] = {"cnt": 0}
        vals_min = {(k, "min"): math.inf for k in self.metrics}
        vals_mass2 = {(k, "mass2"): 0.0 for k in self.metrics}
        vals_mass = {(k, "mass"): 0.0 for k in self.metrics}
        vals_max = {(k, "max"): 0.0 for k in self.metrics}
        self[marker].update(vals_mass2)
        self[marker].update(vals_mass)
        self[marker].update(vals_max)
        self[marker].update(vals_min)

    def calc_metric_stats(self, marker, metric):
        cnt = self[marker]["cnt"]
        mx = self[marker][metric, "max"]
        mn = self[marker][metric, "min"]
        ave = self[marker][metric, "mass"] / cnt
        ems2 = self[marker][metric, "mass2"] / cnt
        if self.metrics[metric]["cv"] and ave > 0:
            cv = 100.0 * math.sqrt(ems2 - ave**2) / ave
        elif self.metrics[metric]["cv"] and ave == 0:
            cv = 0.0
        else:
            cv = None
        return mx, mn, ave, cv

    def repr_metric(self, marker, metric):
        mx, mn, ave, cv = self.calc_metric_stats(marker, metric)
        out = f"\t{metric} maximum: " + self.format_to_print(metric, mx)
        out += f"\n\t{metric} minimum: " + self.format_to_print(metric, mn)
        out += f"\n\t{metric} average: " + self.format_to_print(metric, ave)
        if self.metrics[metric]["cv"] and ave > 0:
            out += f"\n\t{metric} cv: {round(cv, 1)}%"
        elif self.metrics[metric]["cv"] and ave == 0:
            out += f"\n\t{metric} cv: 0%"
        elif self.metrics[metric]["cv"]:
            out += f"\n\t{metric} cv: --"
        return f"{out}\n"

    def __str__(self):
        outstr = "MemorySampler5 summarize:\n"
        for marker in self:
            if marker in ("start", "stop"):
                continue
            count = self[marker]["cnt"]
            outstr += f"Marker: {marker}\n"
            outstr += f"\tsamples: {count}\n"
            if count == 0:
                return outstr
            for metric in self.metrics:
                outstr += self.repr_metric(marker, metric)
        return outstr

    def report_summary(self, chunks, extra):
        out = {
            **extra,
            "chunks_number": chunks,
            "chunk_size": self.chunk_size,
            "metrics": dict(self.metrics),
            "markers": {},
        }
        for marker in self:
            out["markers"][marker] = {}
            out["markers"][marker]["samples"] = self[marker]["cnt"]
            out["markers"][marker]["stats"] = {}
            for metric in self.metrics:
                mx, mn, ave, cv = self.calc_metric_stats(marker, metric)
                out["markers"][marker]["stats"][metric] = {}
                out["markers"][marker]["stats"][metric]["ave"] = self.format_to_export((metric, ave))
                out["markers"][marker]["stats"][metric]["max"] = self.format_to_export((metric, mx))
                out["markers"][marker]["stats"][metric]["min"] = self.format_to_export((metric, mn))
                if cv is None:
                    continue
                out["markers"][marker]["stats"][metric]["cv"] = cv
        return out

    def format_to_export(self, metric_value):
        metric, value = metric_value
        mconf = self.metrics.get(metric)
        formated_value = round(value / mconf["denom"], mconf["digits"])
        return formated_value

    def format_to_print(self, metric, value):
        mconf = self.metrics.get(metric)
        formated_value = round(value / mconf["denom"], mconf["digits"])
        return f"{formated_value} {mconf['unit']}"

    def add_to_summary(self, marker, metric, val):
        self[marker][metric, "max"] = max(self[marker][metric, "max"], val)
        self[marker][metric, "min"] = min(self[marker][metric, "min"], val)
        self[marker][metric, "mass2"] += val**2
        self[marker][metric, "mass"] += val

    def aggregate_and_format(self, marker, values):
        if marker not in self:
            self.init_marker(marker)
        self[marker]["cnt"] += 1
        for metric, value in zip(self.metrics.keys(), values):
            self.add_to_summary(marker, metric, value)
        mapargs = zip(self.metrics.keys(), values)
        return tuple(map(self.format_to_export, mapargs))

    def collect(self, marker):
        parent_process = psutilProcess(self.process_id)
        mem_info = parent_process.memory_full_info()
        rss_mem, uss_mem, priv_mem = mem_info.rss, 0, 0
        if hasattr(mem_info, "uss"):
            uss_mem = mem_info.uss
        if hasattr(mem_info, "private"):
            priv_mem = mem_info.private
        for child_proc in parent_process.children(recursive=True):
            child_mem_info = child_proc.memory_full_info()
            rss_mem += child_mem_info.rss
            if hasattr(child_mem_info, "uss"):
                uss_mem += child_mem_info.uss
            if hasattr(child_mem_info, "private"):
                priv_mem += child_mem_info.private
        sys_mem = psutil.virtual_memory().total - psutil.virtual_memory().available
        nsys_mem = 100 * float(sys_mem) / psutil.virtual_memory().total
        vals5 = rss_mem, uss_mem, priv_mem, sys_mem, nsys_mem
        return self.aggregate_and_format(marker, vals5)


class MemoryMarkerMonitor(list):
    def __init__(self, conn, process_id, sampling_interval, path_prefix):
        self.sampler = MemorySampler5(process_id)
        self.sampling_interval = float(sampling_interval)
        self.process_id = int(process_id)
        self.last_ts = time.perf_counter()
        self.start_perf = time.perf_counter()
        self.start_time_ns = time.time_ns()
        self.marker = "start"
        self.conn = conn

        self.path_prefix = Path(path_prefix)
        if not self.path_prefix.exists():
            self.path_prefix.mkdir(parents=True)
        self.collect_samples()
        self.file_counter = 0

    def deduce_filename(self, report_json=False):
        def fnfunc(path_prefix, process_id, file_counter):
            return path_prefix / f"{process_id}_{file_counter}.txt"

        if report_json:
            return Path(self.path_prefix / f"{self.process_id}_summary.json")
        while fnfunc(self.path_prefix, self.process_id, self.file_counter).exists():
            self.file_counter += 1
        return fnfunc(self.path_prefix, self.process_id, self.file_counter)

    def loop(self, metadata):
        while True:
            if self.conn.poll():
                marker = self.conn.recv()
                self.marker = marker
                if self.marker == "stop":
                    print("Memory worker: Received stop signal")
                    self.collect_samples()
                    print(self.sampler)
                    break
            if len(self) >= self.sampler.chunk_size:
                self.write_chunk()
            try:
                self.collect_samples()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                print("Memory worker: Target process no longer accessible")
                break
            except Exception as e:
                print(f"Error collecting samples: {e}")
            elapsed = time.perf_counter() - self.last_ts
            sleep_time = self.sampling_interval - elapsed
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.001))
        self.write_report(metadata)
        self.write_chunk()

    def write_report(self, extra):
        fname = self.deduce_filename(report_json=True)
        with open(fname, "w", encoding="utf-8") as fd:
            saved_chunks = self.file_counter + 1
            data = self.sampler.report_summary(saved_chunks, extra)
            json.dump(data, fd, indent=2)

    def write_chunk(self):
        if not self:
            return

        counter = len(self)
        fname = self.deduce_filename()
        with open(fname, "w", encoding="utf-8") as fd:
            fd.write(f"#ts marker {self.sampler.header}\n")
             lines = []
             while self:
                 row = self.pop(0)
                 lines.append(" ".join(map(str, row)))
            fd.write("\n".join(lines))
            fd.write(f"\n#number of samples: {counter}\n")

    def collect_samples(self):
        try:
            vals = self.sampler.collect(self.marker)
            self.last_ts = time.perf_counter()

            elapsed_seconds = self.last_ts - self.start_perf
            elapsed_ns = int(elapsed_seconds * 1_000_000_000)
            ts = self.start_time_ns + elapsed_ns
            self.append((ts, self.marker, *vals))
        except Exception as err:
            raise Exception(err)

class MemoryMarkerHandler:
    def __init__(self, args):
        mode = args.memory_consumption
        cooldown = args.memory_consumption_cooldown
        interval = args.memory_consumption_interval
        report_path = args.memory_consumption_dir

        parent_pid = os.getpid()
        self.parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        pargs = child_conn, parent_pid, interval, report_path, mode, cooldown
        self.background_process = mProcess(target=self.background_worker, args=pargs)
        self.background_process.start()
        self.set_high_priority()
        if cooldown is not None:
            self.update_marker("cooldown")
            time.sleep(cooldown)

    def set_high_priority(self):
        priority = psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -10
        try:
            time.sleep(0.1)
            psutilproc = psutilProcess(self.background_process.pid)
            psutilproc.nice(priority)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            print(f"Could not set process priority")
        except PermissionError:
            print(f"No permission to set process priority")

    @staticmethod
    def background_worker(conn, pid, interval, path, mode, cooldown):
        mmm = MemoryMarkerMonitor(conn, pid, interval, path)
        metadata = {
            "mode": mode,
            "cooldown": cooldown,
            "interval": interval,
            "process": pid,
            "path": path,
        }

        try:
            mmm.loop(metadata)
            conn.close()
            log.info("Memory monitor background worker stopped")
        except Exception as e:
            print(f"Error in background worker: {e}")
            traceback.print_exc()

    def update_marker(self, marker):
        try:
            if marker == "stop":
                self.parent_conn.send("stop")
                self.background_process.join()
                self.parent_conn.close()
            else:
                self.parent_conn.send(marker)
        except OSError:
            if marker != "stop":
                log.warning(f"{marker}: markers are not longer accepted!")

    def stop(self):
        try:
            if self.background_process.is_alive():
                self.parent_conn.send("stop")
            self.background_process.join(timeout=5.0)
            if self.background_process.is_alive():
                self.background_process.terminate()
                self.background_process.join(timeout=2.0)
            if self.background_process.is_alive():
                self.background_process.kill()
                self.background_process.join()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                self.parent_conn.close()
            except:
                pass
