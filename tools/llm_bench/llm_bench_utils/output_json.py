# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from llm_bench_utils.memory_monitor import MemoryUnit, MemThreadHandler


def estimate_throughput(latency, bs, ms=True):
    try:
        if latency > 0:
            factor = 1000.0 if ms else 1.0
            return round(int(bs) * factor / latency, 5)
    except ValueError:
        return None
    return None

KEY_MAPPING = {
    "iteration": "iteration",
    "input_size": "input_size",
    "infer_count": "infer_count",
    "output_size": "output_size",
    "generation_time": "generation_time",
    "latency": "latency",
    "result_md5": "result_md5",
    "first_token_latency": "first_latency",
    "other_tokens_avg_latency": "second_avg_latency",
    "first_token_infer_latency": "first_infer_latency",
    "other_tokens_infer_avg_latency": "second_infer_avg_latency",
    "max_rss_mem_consumption": "max_rss_mem",
    "max_rss_mem_increase": "max_increase_rss_mem",
    "max_rss_mem_share": "max_share_rss_mem",
    "max_sys_mem_consumption": "max_sys_mem",
    "max_sys_mem_increase": "max_increase_sys_mem",
    "max_sys_mem_share": "max_share_sys_mem",
    "tokenization_time": "tokenization_time",
    "detokenization_time": "detokenization_time",
    "chat_idx": "chat_idx",
    "prompt_idx": "prompt_idx"
}

def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, iter_timestamp, memory_data_collector):
    metadata = {'model': model, 'framework': framework, 'device': device, 'precision': model_precision,
                'num_beams': model_args['num_beams'], 'batch_size': model_args['batch_size']}
    result = []
    total_iters = len(iter_data_list)
    for i in range(total_iters):
        iter_data = iter_data_list[i]

        result_md5 = []
        for idx_md5 in range(len(iter_data['result_md5'])):
            result_md5.append(iter_data['result_md5'][idx_md5])

        input_idx = iter_data['chat_idx'] if iter_data['chat_idx'] != "" else iter_data['prompt_idx']
        timestamp_start, timestamp_end = get_timestamp(iter_data['iteration'], input_idx, iter_timestamp)
        other_latency = iter_data["other_tokens_avg_latency"]
        res_data = {
            "iteration": iter_data["iteration"],
            "input_size": iter_data["input_size"],
            "infer_count": iter_data["infer_count"],
            "output_size": iter_data["output_size"],
            "second_avg_latency": round(other_latency, 5) if other_latency != "" else other_latency,
            "prompt_idx": iter_data["prompt_idx"],
            "chat_idx": iter_data["chat_idx"],
            "result_md5": result_md5,
            "start": timestamp_start,
            "end": timestamp_end,
        }

        for key in ["latency", "generation_time", "first_token_latency", "first_token_infer_latency", "other_tokens_infer_avg_latency", "tokenization_time", "detokenization_time"]:
            value = round(iter_data[key], 5) if iter_data[key] != "" else iter_data[key]
            json_key = KEY_MAPPING[key]
            res_data[json_key] = value

        # optional metrics
        for key in ["max_rss_mem_consumption", "max_sys_mem_consumption", "max_rss_mem_increase", "max_sys_mem_increase", "max_rss_mem_share", "max_sys_mem_share"]:
            if key in iter_data:
                value = iter_data[key]
                if value is None or value == "":
                    continue

                json_key = KEY_MAPPING[key]
                res_data[json_key] = round(value, 5)

        second_token_throughput = estimate_throughput(other_latency, model_args["batch_size"])
        if second_token_throughput:
            res_data["second_token_throughput"] = second_token_throughput
        result.append(res_data)

    keys_to_average = [
        'generation_time',
        'latency',
        'first_latency',
        'second_avg_latency',
        'first_infer_latency',
        'second_infer_avg_latency',
        'tokenization_time',
        'detokenization_time'
    ]
    results_averaged = {}
    for key in keys_to_average:
        values = [x[key] for x in result[1:] if x[key] != '']
        if len(values) > 0:
            results_averaged[key] = round(sum(values) / len(values), 5)

    if "second_avg_latency" in results_averaged:
        avg_2nd_tokens_latency = results_averaged.get("second_avg_latency")
        second_token_throughput = estimate_throughput(avg_2nd_tokens_latency, model_args["batch_size"])
        if second_token_throughput:
            results_averaged["second_token_throughput"] = second_token_throughput

    output_result = {'metadata': metadata,
                     'perfdata': {'compile_time': pretrain_time,
                                  'results': result} | get_pre_gen_memory_data(memory_data_collector)}
    if len(results_averaged) > 0:
        output_result['perfdata']['results_averaged'] = results_averaged

    with open(report_file, 'w') as outfile:
        json.dump(output_result, outfile, indent=4)


def get_pre_gen_memory_data(memory_data_collector: MemThreadHandler | None, print_unit: MemoryUnit | None = None):
    no_info = MemThreadHandler.MEMORY_NOT_COLLECTED
    suffix = f"({MemThreadHandler.DEF_MEM_UNIT.value})"
    if print_unit is not None:
        suffix = f"({print_unit.value})"
    data = {
        f"initial_sys_mem{suffix}": no_info,
        f"initial_rss_mem{suffix}": no_info,
        f"compile_max_rss_mem{suffix}": no_info,
        f"compile_max_sys_mem{suffix}": no_info,
        f"compile_max_increase_rss_mem{suffix}": no_info,
        f"compile_max_increase_sys_mem{suffix}": no_info,
    }
    if not memory_data_collector:
        return data

    data = data | memory_data_collector.get_initial_mem_data(print_unit)
    data = data | memory_data_collector.get_compilation_mem_data(print_unit)
    return data


def get_timestamp(iter_idx, prompt_idx, iter_timestamp):
    timestamp_start = ''
    timestamp_end = ''
    if iter_idx in iter_timestamp.keys():
        if prompt_idx in iter_timestamp[iter_idx].keys():
            timestamp_start = iter_timestamp[iter_idx][prompt_idx]['start']
            timestamp_end = iter_timestamp[iter_idx][prompt_idx]['end']

    return timestamp_start, timestamp_end
