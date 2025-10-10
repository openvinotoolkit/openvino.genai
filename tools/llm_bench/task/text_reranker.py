# -*- coding: utf-8 -*-
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import json
import torch
import scipy
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
from llm_bench_utils.prompt_utils import get_text_prompt
import llm_bench_utils.gen_output_data as gen_output_data
from task.pipeline_utils import CommonPipeline, execution_time_in_sec
from llm_bench_utils.memory_monitor import MemMonitorWrapper
from pathlib import Path
from typing import Any


FW_UTILS = {"pt": llm_bench_utils.pt_utils, "ov": llm_bench_utils.ov_utils}


class TextRerankerOptimum(CommonPipeline):
    def __init__(self, model: object, tokenizer: object | None, args: dict, model_path: Path, mem_consumption_meter: MemMonitorWrapper):
        super().__init__(model, tokenizer, args, model_path, mem_consumption_meter)
        self.genai = False

        self.texts = get_texts_from_file(args)

        self.top_n = args.get("rerank_top_n")
        self.max_length = args.get("rerank_max_length")
        self.use_case = args.get("use_case")

    # according to transformers Qwen3-Embedding-0.6B model card:
    # https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#transformers-usage
    @execution_time_in_sec
    def tokenize_qwen(self, input_text):
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '\
                 + 'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        task = "Given a web search query, retrieve relevant passages that answer the query"
        max_length = self.max_length or 8192
        pairs = []
        if self.use_case.is_qwen_causallm_arch(self.model.config):
            for doc in self.texts:
                pairs.append(f"<Instruct>: {task}\n<Query>: {input_text}\n<Document>: {doc}")

            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            inputs = self.tokenizer(
                pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for i, ele in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length, padding_side='left').to(self.model.device)
        else:
            for doc in self.texts:
                pairs.append(f"{prefix}<Instruct>: {task}\n<Query>: {input_text}\n<Document>: {doc}{suffix}")
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt", padding_side='left')
        return inputs

    @execution_time_in_sec
    def tokenize(self, input_text: str, **kwargs):
        tokenizer_kwargs = {"truncation": True, "padding": True}
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
        inputs = [input_text] * len(self.texts)
        input_data = self.tokenizer(inputs, self.texts, return_tensors="pt", **tokenizer_kwargs)
        return input_data

    def print_generated(self, iter_num: int, generation_result: Any, prompt_idx: int):
        iter_str = "warm-up" if iter_num == 0 else f"{iter_num}"
        prefix = f"[{iter_str}][P{prompt_idx}]"
        for index, score in generation_result:
            metrics_print.print_unicode(f"{prefix} Document {index}, score: {score:.4f}{': ' + self.texts[index] if iter_num == 0 else ''}")

    @execution_time_in_sec
    def generate(self, input_data: Any, **kwargs):
        with torch.no_grad():
            outputs = self.model(**input_data).logits

        # according to transformers Qwen3-Embedding-0.6B model card:
        # https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#transformers-usage
        if self.use_case.is_qwen_causallm_arch(self.model.config):
            batch_scores = outputs[:, -1, :]

            token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        else:
            if outputs.shape[1] > 1:
                scores = outputs[:, 1]
            else:
                scores = outputs.flatten()
            scores = scipy.special.expit(scores)
        generation_result = []
        for index, (score, _) in enumerate(zip(scores, self.texts)):
            generation_result.append((index, score))

        generation_result.sort(key=lambda x: x[1], reverse=True)
        return generation_result[: self.top_n]

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
        return gen_output_data.embed_iterate_data(
            iter_idx=iter_num,
            in_size=input_token_size * self.batch_size,
            infer_count=infer_count,
            total_time=gen_time_list,
            latency=infer_duration_list,
            max_rss_mem=max_rss_mem_consumption,
            max_rss_mem_increase=rss_mem_increase,
            max_sys_mem=max_sys_mem_consumption,
            max_sys_mem_increase=sys_mem_increase,
            prompt_idx=prompt_index,
            tokenization_time=tokenization_time,
        )

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
        tm_list = []
        tm_infer_list = []
        if bench_hook is not None:
            tm_list = bench_hook.get_time_list()
            log.debug("latency of all texts:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
            tm_infer_list = bench_hook.get_time_infer_list()
            log.debug("latency of all infers:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
        iter_data = self.gen_iterate_data(
            num_input_tokens,
            iter_num,
            len(tm_list),
            generation_time + tokenization_time[0],
            sum(tm_list) * 1000,
            max_rss_mem_consumption,
            rss_mem_increase,
            max_sys_mem_consumption,
            sys_mem_increase,
            prompt_index,
            tokenization_time,
        )
        metrics_print.print_metrics(
            iter_num,
            iter_data,
            tm_list,
            tm_infer_list,
            warm_up=(iter_num == 0),
            tokenization_time=tokenization_time,
            batch_size=self.batch_size,
            prompt_idx=prompt_index,
            latency_unit="prompt",
            text_emb=True,
        )
        self.print_generated(iter_num, generation_result, prompt_index)

        return iter_data, []

    def run(self, input_text: str, iter_num: int, prompt_index: int, proc_id: int, bench_hook: object | None) -> tuple[dict, list]:
        if self.model.config.model_type == "qwen3":
            tokenized_input, tokenization_time = self.tokenize_qwen(input_text)
        else:
            tokenized_input, tokenization_time = self.tokenize(input_text)
        input_tokens = tokenized_input["input_ids"] if "input_ids" in tokenized_input else tokenized_input
        input_token_size = input_tokens[0].numel() * len(self.texts)
        self.print_batch_size_info(iter_num, input_token_size)

        max_rss_mem_consumption = ""
        max_sys_mem_consumption = ""
        rss_mem_increase = ""
        sys_mem_increase = ""
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.start()
        generation_result, generation_time = self.generate(tokenized_input)
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.stop_and_collect_data(f"{'P' + str(iter_num) if iter_num > 0 else 'warm-up'}_{proc_id}")
            max_rss_mem_consumption, rss_mem_increase, max_sys_mem_consumption, sys_mem_increase = self.mem_consumption_meter.get_data()

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
            [tokenization_time * 1000],
            None,
            proc_id,
            bench_hook,
        )

        if bench_hook is not None:
            bench_hook.clear_time_list()
            bench_hook.clear_time_infer_list()

        return iter_data, []


class TextRerankerGenAI(CommonPipeline):
    def __init__(self, model: object, tokenizer: object | None, args: dict, model_path: Path, mem_consumption_meter: MemMonitorWrapper):
        super().__init__(model, tokenizer, args, model_path, mem_consumption_meter)

        if self.batch_size != 1:
            log.warning("Only batch size 1 available for Text Reranker Pipeline")
            self.batch_size = 1

        # documents
        self.texts = get_texts_from_file(args)

        self.top_n = args.get("rerank_top_n")
        self.max_length = args.get("rerank_max_length")

    def tokenize(self, input_text: str, **kwargs):
        tokenizer_kwargs = {"truncation": True, "padding": True}
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
        inputs = [input_text] * len(self.texts)
        input_data = self.tokenizer(inputs, return_tensors="pt", **tokenizer_kwargs)
        input_tokens = input_data["input_ids"] if "input_ids" in input_data else input_data
        return input_tokens

    @execution_time_in_sec
    def generate(self, input_data: Any, **kwargs):
        return self.model.rerank(input_data, self.texts)

    def print_generated(self, iter_num: int, generation_result: Any, prompt_idx: int):
        iter_str = "warm-up" if iter_num == 0 else f"{iter_num}"
        prefix = f"[{iter_str}][P{prompt_idx}]"
        for index, score in generation_result:
            metrics_print.print_unicode(f"{prefix} Document {index}, score: {score:.4f}{': ' + self.texts[index] if iter_num == 0 else ''}")

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
        return gen_output_data.embed_iterate_data(
            iter_idx=iter_num,
            in_size=input_token_size * self.batch_size,
            infer_count=infer_count,
            total_time=gen_time_list,
            latency=infer_duration_list,
            max_rss_mem=max_rss_mem_consumption,
            max_rss_mem_increase=rss_mem_increase,
            max_sys_mem=max_sys_mem_consumption,
            max_sys_mem_increase=sys_mem_increase,
            prompt_idx=prompt_index,
            tokenization_time=tokenization_time,
        )

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
        tm_list = [generation_time]
        tm_infer_list = [generation_time]
        iter_data = self.gen_iterate_data(
            num_input_tokens,
            iter_num,
            len(tm_list),
            generation_time,
            generation_time * 1000,
            max_rss_mem_consumption,
            rss_mem_increase,
            max_sys_mem_consumption,
            sys_mem_increase,
            prompt_index,
            tokenization_time,
        )

        self.print_generated(iter_num, generation_result, prompt_index)
        metrics_print.print_metrics(
            iter_num,
            iter_data,
            tm_list,
            tm_infer_list,
            warm_up=(iter_num == 0),
            tokenization_time=tokenization_time,
            batch_size=self.batch_size,
            prompt_idx=prompt_index,
            latency_unit="prompt",
            text_rerank={"texts_num": len(self.texts)},
        )

        return iter_data, []

    def run(self, input_text: str, iter_num: int, prompt_index: int, proc_id: int, bench_hook: object | None) -> tuple[dict, list]:
        tokenized_input = self.tokenize(input_text)
        input_token_size = tokenized_input[0].numel() * len(self.texts)
        self.print_batch_size_info(iter_num, input_token_size)

        max_rss_mem_consumption = ""
        max_sys_mem_consumption = ""
        rss_mem_increase = ""
        sys_mem_increase = ""
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.start()
        generation_result, generation_time = self.generate(input_text)
        if (self.mem_consumption_level == 1 and iter_num == 0) or self.mem_consumption_level == 2:
            self.mem_consumption_meter.stop_and_collect_data(f"{'P' + str(iter_num) if iter_num > 0 else 'warm-up'}_{proc_id}")
            max_rss_mem_consumption, rss_mem_increase, max_sys_mem_consumption, sys_mem_increase = self.mem_consumption_meter.get_data()

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
            [],
            None,
            proc_id,
            bench_hook,
        )
        return iter_data, []


def run_text_reranker_benchmark(
    model_path: Path, framework: str, device: str, args: dict, num_iters: int, mem_consumption: MemMonitorWrapper
) -> tuple[list, float, dict]:
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_reranker_model(model_path, device, mem_consumption, **args)
    iter_data_list = []
    input_text_list = get_text_prompt(args)
    if args["prompt_index"] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, _ in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args["prompt_index"]:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError("==Failure prompts is empty ==")

    if not use_genai:
        text_reranker_pipeline = TextRerankerOptimum(model, tokenizer, args, model_path, mem_consumption)
    else:
        text_reranker_pipeline = TextRerankerGenAI(model, tokenizer, args, model_path, mem_consumption)

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args["subsequent"] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                iter_data_list.append(launch(text_reranker_pipeline, num, p_idx, iter_timestamp, input_text, proc_id, bench_hook))
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                iter_data_list.append(launch(text_reranker_pipeline, num, p_idx, iter_timestamp, input_text, proc_id, bench_hook))

    metrics_print.print_average(iter_data_list, prompt_idx_list, args["batch_size"], False, True, latency_unit="text")
    return iter_data_list, pretrain_time, iter_timestamp


def launch(pipeline: CommonPipeline, iter_num: int, prompt_idx: int, iter_timestamp: dict, input_text: str, proc_id: int, bench_hook: object | None) -> dict:
    if iter_num == 0:
        metrics_print.print_unicode(
            f"[warm-up][P{prompt_idx}] Input query: {input_text}\n Input texts: {pipeline.texts}", f"[warm-up][P{prompt_idx}] Unable print input text"
        )
    iter_timestamp[iter_num][prompt_idx]["start"] = datetime.datetime.now().isoformat()
    iter_data, _ = pipeline.run(input_text, iter_num, prompt_idx, proc_id, bench_hook)
    iter_timestamp[iter_num][prompt_idx]["end"] = datetime.datetime.now().isoformat()
    prefix = "[warm-up]" if iter_num == 0 else "[{}]".format(iter_num)
    log.info(f"{prefix}[P{prompt_idx}] start: {iter_timestamp[iter_num][prompt_idx]['start']}, end: {iter_timestamp[iter_num][prompt_idx]['end']}")

    return iter_data


def get_texts_from_file(args: dict) -> list:
    texts_list = []
    if args["rerank_texts_file"] is not None and args["rerank_texts"] is not None:
        raise RuntimeError("== --texts and --texts_file are set together, they define lists of texts both, please choose one of them ==")

    if args["rerank_texts_file"] is not None:
        for input_texts_file in args["rerank_texts_file"]:
            if not input_texts_file.endswith(".jsonl"):
                raise RuntimeError(f"== The texts file:{input_texts_file} should be ended with .jsonl ==")
            if not os.path.exists(input_texts_file):
                raise RuntimeError(f"== The texts file:{input_texts_file} does not exist ==")
            log.info(f"Read texts from {input_texts_file}")
            with open(input_texts_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    texts_list.append(data["text"])
    else:
        if args["rerank_texts"] is not None:
            texts_list = args["rerank_texts"]
        else:
            texts_list = [
                "Intel Core Ultra processors incorporate an AI-optimized architecture that supports "
                + "new user experiences and the next wave of commercial applications.",
                "Intel Core Ultra processors are designed to provide enhanced performance and efficiency for a wide range of computing tasks."
            ]
    return texts_list
