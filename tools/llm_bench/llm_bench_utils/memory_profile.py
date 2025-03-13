# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import logging as log

from pathlib import Path

from llm_bench_utils.memory_monitor import MemoryMonitor, MemoryType, MemoryUnit


class MemMonitorWrapper():
    INTERVAL = 0.01

    def __init__(self):
        self.memory_unit = MemoryUnit.MiB
        self.memory_monitor_configurations = [MemoryType.RSS, MemoryType.SYSTEM]

        self.save_dir = None
        self.memory_monitors = {}
        self.memory_data = {}

    def create_monitors(self):
        for memory_type in self.memory_monitor_configurations:
            self.memory_monitors[memory_type] = MemoryMonitor(interval=self.INTERVAL, memory_type=memory_type, memory_unit=self.memory_unit)

    def set_dir(self, dir):
        if not Path(dir).exists():
            log.warning(f"Path to dir for memory consamption data is not exists {dir}, run without it.")
        else:
            self.save_dir = Path(dir)

    def start(self, interval=None, delay=None):
        self.memory_data = {}
        for memory_monitor in self.memory_monitors.values():
            if interval:
                memory_monitor.interval = interval
            memory_monitor.start()

        # compilation could be very fast, apply delay
        if delay:
            time.sleep(delay)
        else:
            sleep_time = interval * 3 if interval else self.INTERVAL * 3
            time.sleep(sleep_time)

    def stop_and_collect_data(self, dir_name='mem_monitor_log'):
        self.stop()

        for memory_type, memory_monitor in self.memory_monitors.items():
            if not memory_monitor._memory_values_queue or len(memory_monitor._memory_values_queue.queue) == 0:
                continue

            time_values, memory_values = memory_monitor.get_data(memory_from_zero=True)
            self.memory_data[memory_type] = max(memory_values)

            if self.save_dir:
                memory_monitor.save_memory_logs(time_values, memory_values, save_dir=self.save_dir / dir_name)
                time_values, memory_values = memory_monitor.get_data(memory_from_zero=False)
                memory_monitor.save_memory_logs(time_values, memory_values, save_dir=self.save_dir / f"{dir_name}_full_proc")

    def stop(self):
        # Stop addition of new values as soon as possible
        for memory_monitor in self.memory_monitors.values():
            memory_monitor._monitoring_thread_should_stop = True

        for memory_monitor in self.memory_monitors.values():
            memory_monitor.stop()
            memory_monitor.interval = self.INTERVAL

    def get_data(self):
        return self.memory_data.get(MemoryType.RSS, -1), self.memory_data.get(MemoryType.SYSTEM, -1)

    def log_data(self, comment):
        max_rss_mem, max_sys_mem = self.get_data()
        msg = (f"Max rss memory cost {comment}: {max_rss_mem:.2f}{self.memory_unit.value}, "
               f"max system memory memory cost {comment}: {max_sys_mem:.2f}{self.memory_unit.value}")
        log.info(msg)
