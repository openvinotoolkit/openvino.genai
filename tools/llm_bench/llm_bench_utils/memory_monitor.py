# Copyright (c) 2025 Intel Corporation
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
import time
from enum import Enum
from functools import lru_cache
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import psutil
import matplotlib
import matplotlib.pyplot as plt
import logging as log


# CUSTOM FIX TO AVOID ISSUE: RuntimeError: main thread is not in main loop
matplotlib.use('Agg')


class MemoryType(Enum):
    RSS = "rss"
    SYSTEM = "system"


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

        :param at_exit_fn: A callable to execute at program exit. Useful fot providing logs saving routine, e.g.
            ```
                at_exit_fn = lambda: memory_monitor.save_memory_logs(*memory_monitor.get_data(), save_dir)
                memory_monitor.start(at_exit_fn=at_exit_fn)
            ```
        """
        if self._monitoring_in_progress:
            log.warning(f"Monitoring was already in progress. MemoryMonitor will be restarted and previous data will be lost for {self.memory_type}.")
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
        log_filepath = save_dir / f"{filename_label}.txt"
        with open(log_filepath, "w") as log_file:
            if len(time_values) == 0:
                log_file.write("No measurements recorded.\nPlease make sure logging duration or interval were enough.")
                return
            for timestamp, memory_usage in zip(time_values, memory_values):
                log_file.write(f"{timestamp} {memory_usage:.3f}\n")

            log_file.writelines(
                [
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
            for line in lines[:-2]:
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
        bytes_used = psutil.Process().memory_info().rss
        if include_child_processes:
            for child_process in psutil.Process().children(recursive=True):
                bytes_used += psutil.Process(child_process.pid).memory_info().rss
        return bytes_used

    def _monitor_memory(self):
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


class MemMonitorWrapper():
    def __init__(self):
        self.save_dir = None

        self.interval = 0.01
        self.memory_unit = MemoryUnit.MiB

        self.memory_types = [MemoryType.RSS, MemoryType.SYSTEM]

        self.memory_monitors = {}
        self.memory_data = {'full_mem': {}, 'from_zero': {}}

    def create_monitors(self):
        for memory_type in self.memory_types:
            self.memory_monitors[memory_type] = MemoryMonitor(
                interval=self.interval, memory_type=memory_type, memory_unit=self.memory_unit
            )

    def set_dir(self, dir):
        if not Path(dir).exists():
            log.warning(f"Path to dir for memory consamption data is not exists {dir}, run without it.")
        else:
            self.save_dir = Path(dir)

    def start(self, delay=None):
        self.memory_data = {'full_mem': {}, 'from_zero': {}}
        for mm in self.memory_monitors.values():
            mm.start()

        # compilation could be very fast, apply delay
        if delay:
            time.sleep(delay)
        else:
            time.sleep(self.interval * 3)

    def stop_and_collect_data(self, dir_name='mem_monitor_log'):
        self.stop()

        for mt, mm in self.memory_monitors.items():
            if not mm._memory_values_queue or len(mm._memory_values_queue.queue) == 0:
                continue

            for from_zero in [False, True]:
                time_values, memory_values = mm.get_data(memory_from_zero=from_zero)

                mm_measure_type = 'from_zero' if from_zero else 'full_mem'
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

    def get_data(self):
        return (self.memory_data['full_mem'].get(MemoryType.RSS, -1), self.memory_data['from_zero'].get(MemoryType.RSS, -1),
                self.memory_data['full_mem'].get(MemoryType.SYSTEM, -1), self.memory_data['from_zero'].get(MemoryType.SYSTEM, -1))


class MemStatus():
    def __init__(self, rss=None, sys=None):
        self.rss = rss
        self.sys = sys


class MemoryDataSummarizer():
    MEMORY_NOT_COLLECTED = ''
    DEF_MEM_UNIT = MemoryUnit.MiB

    def __init__(self, memory_monitor: MemMonitorWrapper):
        self.memory_monitor = memory_monitor
        self.iteration_mem_data = []
        self.compilation_mem_info = {'max_mem': MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED),
                                     'increase_mem': MemStatus(self.MEMORY_NOT_COLLECTED, self.MEMORY_NOT_COLLECTED)}

        self.initial_mem_status = self.log_curent_memory_data(prefix="Start")

    def start(self):
        self.memory_monitor.start()

    def stop_and_collect_data(self, dir_name='mem_monitor_log'):
        self.memory_monitor.stop_and_collect_data(dir_name)

    def get_data(self):
        return self.memory_monitor.get_data()

    def log_data(self, compilation_phase: bool = False):
        max_rss_mem, rss_increase, max_sys_mem, sys_increase = self.memory_monitor.get_data()
        comment = ""
        if compilation_phase:
            comment = "on compilation phase"
            self.compilation_mem_info['max_mem'].sys = max_sys_mem
            self.compilation_mem_info['max_mem'].rss = max_rss_mem
            self.compilation_mem_info['increase_mem'].sys = sys_increase
            self.compilation_mem_info['increase_mem'].rss = rss_increase

        msg = (f"Max RSS memory {comment}: {max_rss_mem:.2f}{self.memory_monitor.memory_unit.value}, "
               f"RSS memory increase {comment}: {rss_increase:.2f}{self.memory_monitor.memory_unit.value}, "
               f"Max System memory {comment}: {max_sys_mem:.2f}{self.memory_monitor.memory_unit.value}, "
               f"System memory increase {comment}: {sys_increase:.2f}{self.memory_monitor.memory_unit.value}")
        log.info(msg)

    def log_curent_memory_data(self, prefix : str = ""):
        mem_status = MemStatus(cast_bytes_to(MemoryMonitor.get_rss_memory(), self.memory_monitor.memory_unit),
                               cast_bytes_to(MemoryMonitor.get_system_memory(), self.memory_monitor.memory_unit))
        log.info(f"{prefix} RSS memory {mem_status.rss:.2f}{self.memory_monitor.memory_unit.value}, "
                 f"{prefix} System memory {mem_status.sys:.2f}{self.memory_monitor.memory_unit.value}")
        return mem_status

    def get_initial_mem_data(self, print_unit: MemoryUnit | None = None):
        suffix = f'({print_unit})' if print_unit else ''
        sys = self.initial_mem_status.sys
        rss = self.initial_mem_status.rss
        if print_unit and print_unit != self.memory_monitor.memory_unit:
            sys = convert_mem_unit(sys, self.memory_monitor.memory_unit, print_unit)
            rss = convert_mem_unit(rss, self.memory_monitor.memory_unit, print_unit)

        return {f'initial_sys_mem{suffix}': round(sys, 5),
                f'initial_rss_mem{suffix}': round(rss, 5)}

    def get_compilation_mem_data(self, print_unit: MemoryUnit | None = None):
        suffix = f'({print_unit.value})' if print_unit else ''
        sys_max = self.compilation_mem_info['max_mem'].sys
        rss_max = self.compilation_mem_info['max_mem'].rss
        sys_increase = self.compilation_mem_info['increase_mem'].sys
        rss_increase = self.compilation_mem_info['increase_mem'].rss
        if print_unit and print_unit != self.memory_monitor.memory_unit:
            sys_max = convert_mem_unit(sys_max, self.memory_monitor.memory_unit, print_unit)
            rss_max = convert_mem_unit(rss_max, self.memory_monitor.memory_unit, print_unit)
            sys_increase = convert_mem_unit(sys_increase, self.memory_monitor.memory_unit, print_unit)
            rss_increase = convert_mem_unit(rss_increase, self.memory_monitor.memory_unit, print_unit)

        return {f'compile_max_rss_mem{suffix}': round(rss_max, 5),
                f'compile_max_sys_mem{suffix}': round(sys_max, 5),
                f'compile_max_increase_rss_mem{suffix}': round(rss_increase, 5),
                f'compile_max_increase_sys_mem{suffix}': round(sys_increase, 5)}


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
