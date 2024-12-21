import csv
import os
from subprocess import Popen, TimeoutExpired, run, call
from statistics import median
import datetime

_model_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/customer_test"

models = {
    # model_name: main_model
    "llama" : [
        "llama-2-7b-chat",
        "CodeLlama-7b-hf",
    ],
    # "opt": ["opt-6.7b"],
    # "phi": ["Phi-3-medium-4k-instruct"],
    # "qwen": ["qwen1.5-7b", ],
    # "starcoder": ["starcoder2-7b"],
    # "dolly-v2": ["dolly-v2-3b","dolly-v2-12b"]
}

element_types = [""]


num_run = range(0, 3)

prompts = [
    # "return 0;",
    "Please, implement the C++ class for Timer to get a durations of other class methids.",
    '''
    The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    '''
]

gen_len = [ 100, 300, 500 ]

devices = [ "CPU" ]

row_desc = [
    "prompt",

    "model",
    "model_type",

    "generation_len",
    "device",

    "sdpa_duration, s",
    "sdpa_prompt_lookup_duration, s",
    "prompt_lookup_duration, s",

    "old_prompt_boost",
    "old_prompt_boost, %",
    "new_prompt_lookup_boost",
    "new_prompt_lookup_boost, %",
    "new_vs_old_prompt_lookup_boost",
    "new_vs_old_prompt_lookup_boost, %",
]


codenames = {
    "total_duration": "Total duration, ms: ",
    "main_duration": "Main model duration, ms: ",
    "main_duration_percentage": "Main model duration, %: ",
    "token_per_second": "Token per sec:",
    "iterations": "Main model iterations: ",
    "avg_acceptance_rate": "AVG acceptance rate, %: ",
    "accepted_token_cnt": "Accepted tokens by draft model: ",
    "accepted_token_rate": "Accepted token rate, %: "
}

binary_path = "/home/efode/repo/openvino/bin/intel64/Release"

replace = "REPLACE_"
cb_binary_name = "prompt_lookup_decoding_lm"
old_binary_name = "prompt_lookup_decoding_lm_sdpa"
llm_binary_name = f"greedy_causal_lm"

log_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/log.log"
csv_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/results_avg.csv"
csv_path_all = "/home/efode/repo/openvino.genai/temp/speculative_decoding/results_all.csv"

with open(csv_path_all, "w", encoding='UTF-8') as csv_all_file:
    csv_writer_all = csv.writer(csv_all_file, dialect="excel")
    csv_writer_all.writerow(row_desc)

    with open(csv_path, "w", encoding='UTF-8') as csv_file:
        csv_writer = csv.writer(csv_file, dialect="excel")
        csv_writer.writerow(row_desc)

        for prompt in prompts:
            for model_name in models:
                for model in models[model_name]:
                    for element_type in element_types:
                        for device in devices:
                            for generation_len in gen_len:
                                avg_sdpa_duration = []
                                avg_sdpa_prompt_lookup_duration = []
                                avg_pa_prompt_lookup_duration = []
                                
                                for run_id in num_run:
                                    sdpa_duration = 0
                                    pl_sdpa_duration = 0
                                    pl_pa_duration = 0

                                    sdpa_exec_path = os.path.join(binary_path, llm_binary_name)
                                    sdpa_pl_exec_path = os.path.join(binary_path, old_binary_name)
                                    pa_pl_exec_path = os.path.join(binary_path, cb_binary_name)

                                    model_path = os.path.join(_model_path, model_name, model, element_type)

                                    if not os.path.exists(model_path):
                                        print(model_path)
                                        continue

                                    command_line_sdpa = f"{sdpa_exec_path} {model_path} \"{prompt}\" {generation_len} > {log_path}"
                                    command_line_sdpa_pl = f"{sdpa_pl_exec_path} {model_path} \"{prompt}\" {generation_len} > {log_path}"
                                    command_line_pa_pl = f"{pa_pl_exec_path} {model_path} \"{prompt}\" {generation_len} > {log_path}"

                                    try:
                                        print(command_line_sdpa)
                                        time_start = datetime.datetime.now()
                                        run(command_line_sdpa, check=True, shell=True)
                                        time_end = datetime.datetime.now()
                                        sdpa_duration = (time_end - time_start).total_seconds()
                                        avg_sdpa_duration.append(sdpa_duration)
                                    except:
                                        pass
                                    
                                    try:
                                        print(command_line_sdpa_pl)
                                        time_start = datetime.datetime.now()
                                        run(command_line_sdpa_pl, check=True, shell=True)
                                        time_end = datetime.datetime.now()
                                        pl_sdpa_duration = (time_end - time_start).total_seconds()
                                        avg_sdpa_prompt_lookup_duration.append(pl_sdpa_duration)
                                    except:
                                        pass

                                    try:
                                        print(command_line_pa_pl)
                                        time_start = datetime.datetime.now()
                                        run(command_line_pa_pl, check=True, shell=True)
                                        time_end = datetime.datetime.now()
                                        pl_pa_duration = (time_end - time_start).total_seconds()
                                        avg_pa_prompt_lookup_duration.append(pl_pa_duration)
                                    except:
                                        pass
                                    
                                    pl_sdpa_boost = sdpa_duration / pl_sdpa_duration
                                    pl_sdpa_boost_percent = pl_sdpa_boost * 100

                                    pl_pa_boost = sdpa_duration / pl_sdpa_duration
                                    pl_pa_boost_percent = pl_pa_boost * 100

                                    sdpa_vs_pa_boost = sdpa_duration / sdpa_duration
                                    sdpa_vs_pa_boost_percent = sdpa_vs_pa_boost * 100

                                    csv_writer_all.writerow([
                                        prompt,
                                        model,
                                        element_type,
                                        generation_len,
                                        device,
                                        sdpa_duration,
                                        pl_sdpa_duration,
                                        pl_pa_duration,
                                        pl_sdpa_boost,
                                        pl_sdpa_boost_percent,
                                        pl_pa_boost,
                                        pl_pa_boost_percent,
                                        sdpa_vs_pa_boost,
                                        sdpa_vs_pa_boost_percent
                                    ])

                                    if run_id == len(num_run) - 1:
                                        avg_sdpa_duration = median(avg_sdpa_duration)
                                        avg_pl_sdpa_duration = median(avg_sdpa_prompt_lookup_duration)
                                        avg_pl_pa_duration = median(avg_pa_prompt_lookup_duration)

                                        pl_sdpa_boost = avg_sdpa_duration / avg_pl_sdpa_duration
                                        pl_sdpa_boost_percent = pl_sdpa_boost * 100

                                        pl_pa_boost = avg_sdpa_duration / avg_pl_pa_duration
                                        pl_pa_boost_percent = pl_pa_boost * 100

                                        sdpa_vs_pa_boost = avg_pl_sdpa_duration / avg_pl_pa_duration
                                        sdpa_vs_pa_boost_percent = sdpa_vs_pa_boost * 100

                                        csv_writer.writerow([
                                            prompt,
                                            model,
                                            element_type,
                                            generation_len,
                                            device,
                                            sdpa_duration,
                                            pl_sdpa_duration,
                                            pl_pa_duration,
                                            pl_sdpa_boost,
                                            pl_sdpa_boost_percent,
                                            pl_pa_boost,
                                            pl_pa_boost_percent,
                                            sdpa_vs_pa_boost,
                                            sdpa_vs_pa_boost_percent
                                        ])


