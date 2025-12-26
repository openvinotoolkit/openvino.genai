# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log

from typing import Any
from pathlib import Path
from transformers import set_seed

# import openvino as ov

import llm_bench_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data

from llm_bench_utils.memory_monitor import MemMonitorWrapper
from llm_bench_utils.hook_forward import StableDiffusionHook
from llm_bench_utils.prompt_utils import get_video_gen_prompt
from task.pipeline_utils import CommonPipeline, execution_time_in_sec, collect_prompts_step, iteration_step

FW_UTILS = {"pt": llm_bench_utils.pt_utils, "ov": llm_bench_utils.ov_utils}

DEFAULT_NUM_FRAMES = 25
DEFAULT_INFERENCE_STEPS = 25
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_FRAME_RATE = 25


def collect_input_args(
    input_param: dict,
    width: int = None,
    height: int = None,
    num_steps: int = None,
    num_frames: int = None,
    frame_rate: int = None,
):
    input_args = {}
    input_args["width"] = input_param.get("width", width or DEFAULT_IMAGE_WIDTH)
    input_args["height"] = input_param.get("height", height or DEFAULT_IMAGE_HEIGHT)
    input_args["num_inference_steps"] = input_param.get("num_steps", num_steps or DEFAULT_INFERENCE_STEPS)
    input_args["num_frames"] = input_param.get("num_frames", num_frames or DEFAULT_NUM_FRAMES)
    input_args["frame_rate"] = input_param.get("frame_rate", frame_rate or DEFAULT_FRAME_RATE)

    guidance_scale = input_param.get("guidance_scale")
    if guidance_scale is not None:
        input_args["guidance_scale"] = guidance_scale
    guidance_rescale = input_param.get("guidance_scale")
    if guidance_rescale is not None:
        input_args["guidance_rescale"] = guidance_rescale
    if "negative_prompt" in input_param:
        input_args["negative_prompt"] = input_param["negative_prompt"]

    return input_args


class TextToVideoOptimum(CommonPipeline):
    def __init__(
        self,
        model: object,
        tokenizer: object | None,
        args: dict,
        model_path: Path,
        mem_consumption_meter: MemMonitorWrapper,
        time_collection_hook: StableDiffusionHook,
    ):
        super().__init__(model, tokenizer, args, model_path, mem_consumption_meter)
        self.genai = False

        self.use_case = args.get("use_case")
        self.num_steps = args.get("num_steps")
        self.num_frames = args.get("num_frames")
        self.frame_rate = args.get("frame_rate")
        self.height = args.get("height")
        self.width = args.get("width")

        self.time_collection_hook = time_collection_hook

    @execution_time_in_sec
    def generate(self, input_data: Any, **kwargs):
        return self.model(input_data, **kwargs).frames

    def get_input_tokens_num(self, prompt: str):
        input_text_list = prompt * self.batch_size
        tokenized_input, _ = self.tokenize(input_text_list)
        input_tokens = tokenized_input["input_ids"] if "input_ids" in tokenized_input else tokenized_input
        return input_tokens[0].numel()

    def print_batch_size_info(self, iter_num: int, input_args: dict):
        out_str = "[warm-up]" if iter_num == 0 else "[{}]".format(iter_num)
        out_str = (
            f"Input params: Batch_size={self.batch_size}, "
            f"steps={self.num_steps}, width={input_args['width']}, "
            f"height={input_args['height']}, frame number={input_args['num_frames']}"
        )
        if input_args.get("guidance_scale"):
            out_str += f", guidance_scale={input_args['guidance_scale']}"
        if input_args.get("guidance_rescale"):
            out_str += f", guidance_rescale={input_args['guidance_rescale']}"
        log.info(out_str)

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
        result_md5_list = []
        for bs_idx in range(self.batch_size):
            llm_bench_utils.output_file.output_gen_video(generation_result[bs_idx], {'batch_size': self.batch_size,
                                                                                     'model_name': self.model_name,
                                                                                     'output_dir': self.output_dir},
                                                         prompt_index, iter_num, bs_idx, proc_id, '.mp4')

        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=num_input_tokens * self.batch_size,
            infer_count=self.num_steps,
            gen_time=generation_time,
            res_md5=result_md5_list,
            max_rss_mem=max_rss_mem_consumption,
            max_rss_mem_increase=rss_mem_increase,
            max_sys_mem=max_sys_mem_consumption,
            max_sys_mem_increase=sys_mem_increase,
            prompt_idx=prompt_index,
        )
        metrics_print.print_metrics(
            iter_num,
            iter_data,
            warm_up=(iter_num == 0),
            batch_size=self.batch_size,
            stable_diffusion=self.time_collection_hook,
            prompt_idx=prompt_index,
        )

        return iter_data, result_md5_list

    def run(self, input_param: dict, iter_num: int, prompt_index: int, proc_id: int, bench_hook: object | None) -> tuple[dict, list]:
        set_seed(self.seed)

        input_args = collect_input_args(input_param, self.width, self.height, self.num_steps, self.num_frames, self.frame_rate)
        input_token_size = self.get_input_tokens_num(input_param["prompt"])
        if input_param.get("negative_prompt"):
            input_token_size += self.get_input_tokens_num(input_param["negative_prompt"])
        org_num_steps = self.num_steps
        self.num_steps = input_args["num_inference_steps"]
        self.print_batch_size_info(iter_num, input_args)

        max_rss_mem_consumption = ""
        max_sys_mem_consumption = ""
        rss_mem_increase = ""
        sys_mem_increase = ""
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.start()
        generation_result, generation_time = self.generate(input_param["prompt"], **input_args)
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.stop_and_collect_data(f"{'P' + str(iter_num) if iter_num > 0 else 'warm-up'}_{proc_id}")
            max_rss_mem_consumption, rss_mem_increase, max_sys_mem_consumption, sys_mem_increase = self.mem_consumption_meter.get_data()

        iter_data = {}
        iter_data, _ = self.postprocess_output_info(
            generation_result,
            generation_time,
            iter_num,
            [],
            input_token_size,
            max_rss_mem_consumption,
            rss_mem_increase,
            max_sys_mem_consumption,
            sys_mem_increase,
            prompt_index,
            None,
            None,
            proc_id,
            bench_hook,
        )
        self.num_steps = org_num_steps
        self.time_collection_hook.clear_statistics()
        return iter_data, []


def run_video_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    text_list, prompt_idx_list = collect_prompts_step(args, get_video_gen_prompt)

    # If --static_reshape is specified, we need to get width, height, and guidance scale to drop into args
    # as genai's create_image_gen_model implementation will need those to reshape the pipeline before compile().
    if args.get("static_reshape", False):
        input_args = collect_input_args(text_list[0], args["width"], args["height"], args["num_steps"], args["num_frames"], args["frame_rate"])
        args |= input_args

    pipe, pretrain_time, use_genai = FW_UTILS[framework].create_video_gen_model(model_path, device, mem_consumption, **args)
    iter_data_list = []

    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(text_list)}, prompt idx: {prompt_idx_list}')

    if use_genai:
        log.info('not supported yet')
        return [], 0, {}
    else:
        stable_diffusion_hook = StableDiffusionHook()
        if framework == "ov":
            stable_diffusion_hook.init_custom_pipe(pipe)
        image_gen_pipeline = TextToVideoOptimum(pipe, pipe.tokenizer, args, model_path, mem_consumption, stable_diffusion_hook)

    iter_data_list, iter_timestamp = iteration_step(image_gen_pipeline, num_iters, text_list, prompt_idx_list, bench_hook=None, subsequent=args['subsequent'])

    metrics_print.print_average(iter_data_list, prompt_idx_list, args["batch_size"], False)
    return iter_data_list, pretrain_time, iter_timestamp
