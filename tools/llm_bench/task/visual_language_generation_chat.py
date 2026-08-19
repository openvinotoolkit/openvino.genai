# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import hashlib
import datetime
import logging as log
import numpy as np

from llm_bench_utils.prompt_utils import get_vlm_prompt
import llm_bench_utils.output_file
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.prompt_utils import extract_prompt_data
from task.text_generation_chat import (
    get_kv_axes_pos,
    OptimumTextGenerationChatAdapter,
    GenAITextGenerationChatAdapter,
    ChatIterationResult,
    ChatGenerationAdapter,
    update_chat_iteration_with_memory_info,
)
from task.text_generation import (
    save_input_data_to_file,
    print_generated_output,
)

from inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING

from transformers import __version__, set_seed
from packaging.version import Version

TRANSFORMERS_VERSION = Version(__version__)


DEFAULT_OUTPUT_TOKEN_SIZE = 512

FW_UTILS = {"pt": llm_bench_utils.pt_utils, "ov": llm_bench_utils.ov_utils}

FULL_CHAT_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "llava",
    "phi4mm",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "llava_next",
    "llava-qwen2",
    "qwen3_omni",
    "qwen3_omni_moe",
}


class OptimumVLMGenerationChatAdapter(OptimumTextGenerationChatAdapter):
    def __init__(
        self,
        model: object,
        tokenizer: object,
        processor: object,
        config: object,
        args: dict,
    ):
        self.is_genai = False
        self.model = model
        self.language_model = self.model.language_model if hasattr(self.model, "language_model") else model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.args = args

        self.crop_question = (
            "internvl" in config.model_type
            or "minicpmv" in config.model_type
            or "minicpmo" in config.model_type
            or "videochat_flash_qwen" in config.model_type
        )

        self.kv_axes_pos = 2
        if "optimum" in str(type(model)):
            language_model = model.language_model if hasattr(model, "language_model") else model
            self.kv_axes_pos = get_kv_axes_pos(language_model.model)

        if self.config.model_type not in MODEL_TYPE_TO_CLS_MAPPING:
            raise ValueError(
                f"llm_bench doesn't support models with type '{self.config.model_type}' to run in chat mode."
            )

        self.inputs_processor = MODEL_TYPE_TO_CLS_MAPPING[self.config.model_type](chat_mode=True, model=model)

        self.max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
        self.full_chat = (
            args.get("full_chat")
            or "transformers" in str(type(model))
            or (model and self.config.model_type in FULL_CHAT_MODEL_TYPES)
        )

        self.generation_args = None
        self.chat_history = []
        self.past_key_values = None
        self.tokenized_history: list = []

    def init_chat(self):
        self.clear_chat()
        self.generation_args = model_utils.setup_gen_config_use_custom_args()

    def clear_chat(self):
        self.generation_args = None
        self.chat_history = []
        self.past_key_values = None
        self.tokenized_history: list = []

    def configure_past_key_values_for_generation(self, full_input_ids: torch.Tensor, prefix_len: int) -> dict:
        past_key_kwargs = {}
        if "transformers" in str(type(self.model)):
            past_key_kwargs["past_key_values"] = self.past_key_values
        else:
            # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
            # to avoid taking into account past_key_values, will set it to [None]
            states = self.language_model.request.query_state()
            shape = None
            for state in states:
                old_tensor = state.state
                # [BATCH_SIZE, num_kv_heads, seq_len, head_size]
                data = np.array(old_tensor.data)
                shape = data.shape
                break

            if prefix_len > 0:
                fake_past_key_values = ((np.ones(shape, dtype=data.dtype), np.ones(shape, dtype=data.dtype)),)
                if TRANSFORMERS_VERSION > Version("5.2.0"):
                    from transformers.cache_utils import DynamicCache

                    fake_layer = (
                        torch.from_numpy(np.ones(shape, dtype=data.dtype)),
                        torch.from_numpy(np.ones(shape, dtype=data.dtype)),
                    )
                    fake_past_key_values = DynamicCache(ddp_cache_data=(fake_layer,))
                past_key_kwargs["past_key_values"] = fake_past_key_values
            else:
                past_key_kwargs["past_key_values"] = None

        return past_key_kwargs

    def run_chat_iteration(
        self, prompt: str, prefix: str, images, videos, audios, bench_hook: object
    ) -> ChatIterationResult:
        set_seed(self.args["seed"])

        self.chat_history.append({"role": "user", "content": prompt})

        # ===== Tokenization =====
        tok_encode_start = time.perf_counter()
        preprocess_inputs = self.inputs_processor.preprocess_inputs(
            prompt,
            images,
            processor=self.processor,
            tokenizer=self.tokenizer,
            config=self.config,
            video=videos,
            audio=audios,
        )
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000

        full_input_ids = preprocess_inputs["input_ids"]
        full_token_list = full_input_ids[0].tolist()

        if not self.full_chat and len(self.tokenized_history) > 0:
            prefix_len = self.find_common_prefix_length(full_token_list)
            if prefix_len < len(self.tokenized_history):
                self.past_key_values = self.trim_kv_cache(prefix_len, self.language_model)
            self.tokenized_history = self.tokenized_history[:prefix_len]
        else:
            prefix_len = 0
            self.past_key_values = None

        generate_kwargs = {}
        if self.past_key_values is not None:
            preprocess_inputs = self.inputs_processor.align_inputs_with_cache(
                self.model, preprocess_inputs, full_input_ids, prefix_len
            )
            generate_kwargs = self.configure_past_key_values_for_generation(full_input_ids, prefix_len)

        model_type = self.config.model_type
        if model_type not in ["phi4mm"]:
            generate_kwargs["tokenizer"] = self.tokenizer

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        result = self.model.generate(
            **preprocess_inputs,
            max_new_tokens=int(self.max_gen_tokens),
            num_beams=self.args["num_beams"],
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
            **self.generation_args,
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start

        if isinstance(result, tuple) and isinstance(result[0], list) and isinstance(result[0][0], str):
            # Some models return a decoded result, like miniCPM-o
            # The output tuple has format (<list of decoded outputs without question/prompt>, <GenerateDecoderOnlyOutput>)
            result = result[1]

        tokens = result.sequences
        if self.crop_question:
            input_ids_len = full_input_ids.shape[-1]
            tokens = tokens[:, input_ids_len:]

        new_past_key_values = getattr(result, "past_key_values", None) if result is not None else None
        if new_past_key_values is not None and not self.full_chat:
            self.past_key_values = new_past_key_values
            generated_ids_list = tokens[0].tolist()
            self.tokenized_history = full_token_list + generated_ids_list
            actual_cache_len = self.get_kv_cache_seq_len(self.language_model)
            if actual_cache_len > 0 and len(self.tokenized_history) != actual_cache_len:
                self.tokenized_history = self.tokenized_history[:actual_cache_len]
        else:
            self.past_key_values = None
            self.tokenized_history = []

        input_token_size = full_input_ids[0].numel()
        num_new_token_input_size = input_token_size - prefix_len
        generated_token_size = len(tokens)

        if generated_token_size > self.max_gen_tokens:
            log.error("Output token size is over max output token size!")

        # ===== Detokenization =====
        tok_decode_start = time.perf_counter()
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        tok_decode_end = time.perf_counter()
        tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
        self.chat_history.append({"role": "assistant", "content": generated_text[0]})
        self.inputs_processor.update_chat_history_with_answer(generated_text[0])

        # ===== Performance Data Collection and Print Results =====
        rendered_chat = self.tokenizer.apply_chat_template(
            self.chat_history, tokenize=False, add_generation_prompt=True
        )

        tm_list = []
        tm_infer_list = []
        if bench_hook is not None:
            tm_list = list(bench_hook.get_time_list())
            log.debug("latency of all tokens:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
            tm_infer_list = bench_hook.get_time_infer_list()
            log.debug("latency of all infers:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
            if self.args["num_beams"] == 1 and generated_token_size != len(tm_infer_list):
                log.warning(
                    f"Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})"
                )

        return ChatIterationResult(
            input_size=num_new_token_input_size,
            output_size=generated_token_size,
            generation_time=generation_time,
            infer_count=len(tm_infer_list),
            tm_list=tm_list,
            tm_infer_list=tm_infer_list,
            tokenization_time=(tok_encode_time, tok_decode_time),
            rendered_chat=rendered_chat,
        )

    def get_messages(self):
        return self.chat_history


class GenAIVLMGenerationChatAdapter(GenAITextGenerationChatAdapter):
    def __init__(self, model: object, args: dict):
        self.is_genai = True
        self.args = args
        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.full_chat = args.get("full_chat", False)
        self.max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]

        self.gen_config = None
        self.chat_history = None
        self.chat_token_size = 0

    def setup_generation_config(self):
        gen_config = self.model.get_generation_config()
        gen_config.max_new_tokens = self.max_gen_tokens
        gen_config.num_beams = self.args["num_beams"]
        gen_config.do_sample = False
        gen_config.ignore_eos = True
        if self.args["pruning_ratio"] is not None:
            gen_config.pruning_ratio = self.args["pruning_ratio"]
        if self.args["relevance_weight"] is not None:
            gen_config.relevance_weight = self.args["relevance_weight"]
        if self.args.get("draft_model", ""):
            from task.text_generation import apply_sd_generation_config

            apply_sd_generation_config(self.args, gen_config)

        return gen_config

    def init_chat(self):
        import openvino_genai

        self.clear_chat()
        self.gen_config = self.setup_generation_config()
        self.chat_history = openvino_genai.ChatHistory()

    def clear_chat(self):
        self.gen_config = None
        self.chat_history = None
        self.chat_token_size = 0

    def run_chat_iteration(
        self, prompt: str, prefix: str, images: list, videos: list, audios: list, bench_hook: object
    ) -> ChatIterationResult:
        self.chat_history.append({"role": "user", "content": prompt})

        kwargs = {}
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            kwargs["audios"] = audios

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        generation_result = self.model.generate(self.chat_history, generation_config=self.gen_config, **kwargs)
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start

        self.chat_history.append({"role": "assistant", "content": generation_result.texts[0]})

        # ==== Performance Data Collection and Print Results =====
        perf_metrics = generation_result.perf_metrics
        num_input_size, num_generated_tokens = self.collect_input_output_sizes(perf_metrics, prefix)
        tm_list, inference_durations, tokenization_time, cache_usage = self.collect_time_perf_metrics(perf_metrics)

        rendered_chat = self.tokenizer.apply_chat_template(self.chat_history, add_generation_prompt=True)

        return ChatIterationResult(
            input_size=num_input_size,
            output_size=num_generated_tokens,
            generation_time=generation_time,
            infer_count=len(tm_list),
            tm_list=tm_list,
            tm_infer_list=inference_durations,
            tokenization_time=tokenization_time,
            rendered_chat=rendered_chat,
            cache_usage=cache_usage,
        )

    def get_messages(self):
        if self.chat_history is None:
            return []
        return self.chat_history.get_messages()


# ===== Common Utils =====
def get_chat_input_data(input_text: list, args: dict):
    # if prompts are set as list, let's use it
    # if prompt is set as single string, let's create chat where prompt will repeat chat_iter times
    input_data = input_text
    if args.get("chat_iter"):
        if len(input_text) == 1:
            input_data = input_text * args["chat_iter"]
        else:
            log.warning(
                f"Chat mode is enabled and chat_iter is {args['chat_iter']}, but input data is set as list."
                "`chat_iter` will be ignored. Chat will be run based on the provided list."
            )
    return input_data


def run_visual_language_generation_chat_common(
    pipeline: ChatGenerationAdapter,
    input_data,
    iter_num,
    args,
    iter_data_list,
    md5_list,
    chat_index,
    bench_hook,
    model_precision,
    proc_id,
    mem_consumption,
):
    if args["batch_size"] != 1:
        log.warning("Batch size is not applicable for VLM chat scenario. Parameter will be ignored and set to 1.")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_data, args)
    save_input_data_to_file(
        [i["prompt"] for i in input_data], args, model_precision, chat_index, iter_num, proc_id, is_chat=True
    )

    # ===== Prepare Config, Additional Args and Chat Managing Variables =====
    pipeline.init_chat()

    prefix = f"[warm-up][C{chat_index}]" if iter_num == 0 else f"[{iter_num}][C{chat_index}]"

    chat_iter_data_list = []
    mem_consumption.start(iter_num)
    for prompt_index, turn_data in enumerate(input_data):
        prompts, images, videos, audios = extract_prompt_data([turn_data], args["video_frames"], pipeline.is_genai)

        log.info(f"{prefix}[P{prompt_index}] Input image nums: {len(images)}")
        log.info(f"{prefix}[P{prompt_index}] Input video nums: {len(videos)}")
        log.info(f"{prefix}[P{prompt_index}] Input audio nums: {len(audios)}")

        chat_iteration_result = pipeline.run_chat_iteration(prompts[0], prefix, images, videos, audios, bench_hook)

        result_md5_list = []
        result_md5_list.append(
            hashlib.new("md5", chat_iteration_result.rendered_chat.encode(), usedforsecurity=False).hexdigest()
        )
        if len(md5_list[iter_num]) == 0:
            md5_list[iter_num] = {}
        if chat_index not in md5_list[iter_num]:
            md5_list[iter_num][chat_index] = {}
        md5_list[iter_num][chat_index][prompt_index] = result_md5_list
        print_generated_output(
            prompt_index,
            iter_num,
            result_md5_list,
            md5_list,
            ["\n" + chat_iteration_result.rendered_chat],
            enable_prompt_permutations=False,
            chat_prompts_num=len(input_data),
            chat_idx=chat_index,
        )

        per_token_time = ""
        if chat_iteration_result.output_size > 0:
            per_token_time = chat_iteration_result.generation_time * 1000 / chat_iteration_result.output_size
        else:
            log.warning("No generated tokens")

        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=chat_iteration_result.input_size,
            infer_count=chat_iteration_result.infer_count,
            out_size=chat_iteration_result.output_size,
            gen_time=chat_iteration_result.generation_time,
            latency=per_token_time,
            res_md5=result_md5_list,
            prompt_idx=prompt_index,
            tokenization_time=chat_iteration_result.tokenization_time,
            chat_idx=chat_index,
        )
        chat_iter_data_list.append(iter_data)

        metrics_print.print_metrics(
            iter_num,
            iter_data,
            chat_iteration_result.tm_list,
            chat_iteration_result.tm_infer_list,
            warm_up=(iter_num == 0),
            tokenization_time=chat_iteration_result.tokenization_time,
            batch_size=args["batch_size"],
            prompt_idx=prompt_index,
            cb_metric=chat_iteration_result.cache_usage,
            chat_idx=chat_index,
        )

        if bench_hook is not None:
            bench_hook.clear_time_list()
            bench_hook.clear_time_infer_list()

    memory_metrics = mem_consumption.iter_stop_and_collect_data(iter_num)
    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)
    metrics_print.print_memory_info(iter_num, chat_iter_data_list[-1], chat_index)

    # === Save perf data ===
    iter_data_list.extend(chat_iter_data_list)

    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(
            pipeline.get_messages(),
            args,
            model_precision,
            chat_index,
            iter_num,
            batchsize_idx=0,
            proc_id=proc_id,
            is_chat=True,
        )

    pipeline.clear_chat()


def run_visual_language_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    mem_consumption.update_marker("model")
    outs = FW_UTILS[framework].create_image_text_gen_model(model_path, device, mem_consumption, **args)
    mem_consumption.activate_cooldown("after model compilation")
    model, processor_config, pretrain_time, bench_hook, use_genai = outs
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num: {} for num in range(num_iters + 1)}
    input_chat_list = get_vlm_prompt(args)

    iter_data_list = []
    md5_list = {num: {} for num in range(num_iters + 1)}

    if args["prompt_index"] is None:
        input_idx_list = [idx for idx, _ in enumerate(input_chat_list)]
        chat_list = input_chat_list
    else:
        input_idx_list = []
        chat_list = []
        for i in args["prompt_index"]:
            if 0 <= i < len(input_chat_list):
                chat_list.append(input_chat_list[i])
                input_idx_list.append(i)

    if len(chat_list) == 0 or any(len(chat_turns) == 0 for chat_turns in chat_list):
        raise RuntimeError("==Failure prompts is empty ==")

    log.info(
        f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
        f"chat nums: {len(chat_list)}, chat idx: {input_idx_list}"
    )

    if use_genai:
        pipeline = GenAIVLMGenerationChatAdapter(model=model, args=args)
    else:
        pipeline = OptimumVLMGenerationChatAdapter(
            model=model,
            tokenizer=processor_config["tokenizer"],
            processor=processor_config["processor"],
            config=processor_config["config"],
            args=args,
        )

    proc_id = os.getpid()
    iter_alias = "C"
    iter_timestamp = model_utils.init_timestamp(num_iters, chat_list, input_idx_list)

    if args["subsequent"] is False:
        for num in range(num_iters + 1):
            for idx, chat_turns in enumerate(chat_list):
                chat_idx = input_idx_list[idx]
                mem_consumption.update_marker(f"step-{num}-{chat_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{chat_idx}] Input text: {chat_turns}",
                        f"[warm-up][{iter_alias}{chat_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][chat_idx]["start"] = datetime.datetime.now().isoformat()
                run_visual_language_generation_chat_common(
                    pipeline,
                    chat_turns,
                    num,
                    args,
                    iter_data_list,
                    md5_list,
                    chat_idx,
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][chat_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][{iter_alias}{chat_idx}]" if num == 0 else f"[{num}][{iter_alias}{chat_idx}]"
                log.info(
                    f"{prefix} start: {iter_timestamp[num][chat_idx]['start']}, end: {iter_timestamp[num][chat_idx]['end']}"
                )
    else:
        for idx, chat_turns in enumerate(chat_list):
            chat_idx = input_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{chat_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{chat_idx}] Input text: {chat_turns}",
                        f"[warm-up][{iter_alias}{chat_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][chat_idx]["start"] = datetime.datetime.now().isoformat()
                run_visual_language_generation_chat_common(
                    pipeline,
                    chat_turns,
                    num,
                    args,
                    iter_data_list,
                    md5_list,
                    chat_idx,
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][chat_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][{iter_alias}{chat_idx}]" if num == 0 else f"[{num}][{iter_alias}{chat_idx}]"
                log.info(
                    f"{prefix} start: {iter_timestamp[num][chat_idx]['start']}, end: {iter_timestamp[num][chat_idx]['end']}"
                )

    metrics_print.print_average(iter_data_list, input_idx_list, args["batch_size"], True, chat_mode=True)
    return iter_data_list, pretrain_time, iter_timestamp
