# -*- coding: utf-8 -*-
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import time
import logging as log
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.memory_monitor import MemMonitorWrapper
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod


def execution_time_in_sec(func):
    def time_wrapper(self, args, **kwargs):
        start = time.perf_counter()
        result = func(self, args, **kwargs)
        end = time.perf_counter()
        exec_time_ms = end - start
        return result, exec_time_ms

    return time_wrapper


class CommonPipeline(ABC):
    DEFAULT_OUTPUT_TOKEN_SIZE = 512

    def __init__(self, model, tokenizer, model_args: dict, model_path: Path, mem_consumption_meter: MemMonitorWrapper):
        self.genai = True

        self.model = model
        self.tokenizer = tokenizer

        self.model_name = model_args["model_name"]
        self.batch_size = model_args["batch_size"]
        self.output_dir = model_args["output_dir"]
        self.infer_count = model_args["infer_count"]
        self.max_gen_tokens = self.DEFAULT_OUTPUT_TOKEN_SIZE if self.infer_count is None else self.infer_count
        self.seed = model_args["seed"]
        self.num_beams = model_args["num_beams"]
        self.model_precision = model_utils.get_model_precision(model_path.parts)

        self.mem_consumption_meter = mem_consumption_meter
        self.mem_consumption_level = model_args.get("mem_consumption", 0)

    @execution_time_in_sec
    def tokenize(self, input_text_list: list, **kwargs):
        """Preprocessing of input text data.

        Args:
            input_text_list (list): Input list of strings for tokenization.

        Returns:
            Tensor | dict | list: Tokenized input.
            Float: Time of tokenizer run.
        """
        if self.genai:
            return self.tokenizer.encode(input_text_list, **kwargs)
        else:
            return self.tokenizer(input_text_list, return_tensors="pt", **kwargs)

    @execution_time_in_sec
    def generate(self, input_data: Any, **kwargs):
        """Run generation of input_data.

        Args:
            input_data (Any): Input data for generate.

        Returns:
            Any: Generation output.
            Float: Generation time.
        """
        return self.model.generate(input_data, **kwargs)

    @execution_time_in_sec
    def detokenize(self, generated_tokens: list, **kwargs):
        """Postprocessing of generated text data.

        Args:
            generated_tokens (list): Input list of strings for tokenization.

        Returns:
            string | list: Text output.
            Float: Detokenization time.
        """
        if self.genai:
            return self.tokenizer.decode(generated_tokens, **kwargs)
        else:
            return self.tokenizer.batch_decode(generated_tokens)

    def print_batch_size_info(self, iter_num: int, num_input_tokens: int):
        """
        Args:
            iter_num (int): Iteration number.
            num_input_tokens (int): Amount of input tokens.
        """
        if self.batch_size <= 1:
            return
        out_str = "[warm-up]" if iter_num == 0 else "[{}]".format(iter_num)
        out_str += " Batch_size={}, ".format(self.batch_size)
        out_str += "all input token size after padding: {} * {}, ".format(num_input_tokens, self.batch_size)
        if self.infer_count is not None:
            out_str += "all max_output_token_size: {} * {}".format(self.infer_count, self.batch_size)
        log.info(out_str)

    def print_generated(self, iter_num: int, generation_result: Any, prompt_idx: int):
        """
        Args:
            iter_num (int): Iteration number.
            generation_result (Any): Output of generation.
            prompt_idx (int): Number of the prompt being processed.
        """
        raise NotImplementedError

    @abstractmethod
    def gen_iterate_data(
        self,
        input_token_size: int,
        iter_num: int,
        infer_count: int,
        gen_time_list: list,
        infer_duration_list: list,
        max_rss_mem_consumption: float,
        rss_mem_increase: float,
        max_sys_mem_consumption: float,
        sys_mem_increase: float,
        prompt_index: int,
        tokenization_time: list,
    ):
        """
        Args:
            input_token_size (int): Amoutn of input tokens.
            iter_num (int): Iteration number.
            infer_count (int): Amount of run inference.
            gen_time_list (list): List of generations duratrion times.
            infer_duration_list (list): List of inferences duration times.
            max_rss_mem_consumption (float): Peak RSS memory consumed during generation
            rss_mem_increase (float): Increase of RSS memory during generation.
            max_sys_mem_consumption (float): Peak System memory consumed during generation
            sys_mem_increase (float): Increase of System memory during generation.
            prompt_index (int): Number of the prompt being processed.
            tokenization_time (list): List of tokenizer/detokenizer run times.

        Returns:
            dict: Collected statistic about launch.
        """
        return {}

    @abstractmethod
    def postprocess_output_info(
        self,
        generation_result: Any,
        generation_time: float,
        iter_num: int,
        input_tokens: list,
        num_input_tokens: int,
        max_rss_mem_consumption: float,
        rss_mem_increase: float,
        max_sys_mem_consumption: float,
        sys_mem_increase: float,
        prompt_index: int,
        tokenization_time: list,
        prev_md5: int,
        proc_id: int,
        bench_hook: object | None,
    ):
        """Collect and print results and statistic information.

        Args:
            generation_result (Any): Generation output.
            generation_time (float): Generation time.
            iter_num (int): Iteration number.
            input_tokens (list): Tokenized input.
            num_input_tokens (int): Amount of input tokens.
            max_rss_mem_consumption (float): Peak RSS memory consumed during generation
            rss_mem_increase (float): Increase of RSS memory during generation.
            max_sys_mem_consumption (float): Peak System memory consumed during generation
            sys_mem_increase (float): Increase of System memory during generation.
            prompt_index (int): Number of the prompt being processed.
            tokenization_time (list): List of tokenizer/detokenizer run times.
            prev_md5 (int): md5 of previous generation.
            proc_id (int): current PID.
            bench_hook (object | None): Class for collection of statistic.

        Returns:
            dict: Collected statistic about launch.
            list: results md5 list.
        """
        return {}, []

    @abstractmethod
    def run(self, input_text: str, iter_num: int, prompt_index: int, proc_id: int, bench_hook: object | None) -> tuple[dict, list]:
        """Run pipeline.

        Args:
            input_text (str): input prompt.
            iter_num (int): Iteration number.
            prompt_index (int): Number of the prompt being processed.
            proc_id (int): current PID.
            bench_hook (object | None): Class for collection of statistic.

        Returns:
            dict: Collected statistic about launch.
            list: results md5 list.
        """
        return {}, []
