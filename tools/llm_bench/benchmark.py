# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import logging as log
import llm_bench_utils.model_utils
from openvino.runtime import get_version
import torch
import traceback
from llm_bench_utils.memory_profile import MemConsumption
import llm_bench_utils.output_csv
import llm_bench_utils.output_json
import task.text_generation as bench_text
import task.image_generation as bench_image
import task.super_resolution_generation as bench_ldm_sr
import task.speech_to_text_generation as bench_speech

DEFAULT_TORCH_THREAD_NUMS = 16
mem_consumption = MemConsumption()


def num_iters_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError('Minimum input value is 0')
    return x


def num_infer_count_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError('Minimum input value is 1')
    return x


def get_argprser():
    parser = argparse.ArgumentParser('LLM benchmarking tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='model folder including IR files or Pytorch files', required=TabError)
    parser.add_argument('-d', '--device', default='cpu', help='inference device')
    parser.add_argument('-r', '--report', help='report csv')
    parser.add_argument('-rj', '--report_json', help='report json')
    parser.add_argument('-f', '--framework', default='ov', help='framework')
    parser.add_argument('-p', '--prompt', default=None, help='one prompt')
    parser.add_argument('-pf', '--prompt_file', nargs='+', default=None,
                        help='Prompt file(s) in jsonl format. Multiple prompt files should be separated with space(s).')
    parser.add_argument('-pi', '--prompt_index', nargs='+', type=num_iters_type, default=None,
                        help='Run the specified prompt index. You can specify multiple prompt indexes, separated by spaces.')
    parser.add_argument('--media', default=None, help='Media file path for speech or visual models.')
    parser.add_argument(
        '-ic',
        '--infer_count',
        default=None,
        type=num_infer_count_type,
        help='set the output token size, the value must be greater than 0.'
    )
    parser.add_argument(
        '-n',
        '--num_iters',
        default=0,
        type=num_iters_type,
        help='number of benchmarking iterations, '
        'if the value is greater than 0, the average numbers exclude the first(0th) iteration,\n'
        'if the value equals 0 (default), execute the warm-up iteration(0th iteration).',
    )
    parser.add_argument('-i', '--images', default=None, help='test images for vision tasks. Can be directory or path to single image')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='specific random seed to generate fix result. Default 42.')
    parser.add_argument(
        '-lc',
        '--load_config',
        default=None,
        required=False,
        help='path to JSON file to load customized configurations.\n'
        'Example for OpenVINO: {\"INFERENCE_NUM_THREADS\":32,\"PERFORMANCE_HINT\":\"LATENCY\"}.\n'
        'Example for Pytorch: {\"PREC_BF16\":true}. Pytorch currently only supports bf16 settings.\n',
    )
    parser.add_argument(
        '-mc',
        '--memory_consumption',
        default=0,
        required=False,
        type=int,
        help='if the value is 1, output the maximum memory consumption in warm-up iterations. If the value is 2,'
        ' output the maximum memory consumption in all iterations.',
    )
    parser.add_argument('-bs', '--batch_size', type=int, default=1, required=False, help='Batch size value')
    parser.add_argument(
        '--fuse_decoding_strategy',
        action='store_true',
        help='Add decoding postprocessing for next token selection to the model as an extra ops. Original hf_model.generate function will be patched.',
    )
    parser.add_argument(
        '--save_prepared_model',
        default=None,
        help='Path to .xml file to save IR used for inference with all pre-/post processing included',
    )
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams in the decoding strategy, activates beam_search if greater than 1')
    parser.add_argument(
        '--torch_compile_backend',
        default=None,
        required=False,
        help='Enables running the torch.compile() with specified backend: pytorch or openvino (default)',
    )
    parser.add_argument(
        '--torch_compile_dynamic',
        action='store_true',
        help='Enables dynamic shape tracking for torch.compile()',
    )
    parser.add_argument(
        '--torch_compile_options',
        default=None,
        required=False,
        help='Options for torch.compile() in JSON format',
    )
    parser.add_argument(
        '--torch_compile_input_module',
        default=None,
        required=False,
        help='Specifies the module to decorate with torch.compile(). By default, parent module will be decorated.',
    )
    parser.add_argument(
        '--convert_tokenizer', action='store_true', help='Convert tokenizer to OpenVINO format'
    )
    parser.add_argument(
        '--subsequent',
        action='store_true',
        help='if the value is True, input prompts are processed in subsequent manner'
        'if the value is False (default), input prompts are processed in interleave manner'
    )
    parser.add_argument('-od', '--output_dir', help='Save the input text and generated text, images to files')
    llm_bench_utils.model_utils.add_stateful_model_arguments(parser)
    parser.add_argument("--genai", action="store_true", help="[DEPRECATED] Use OpenVINO GenAI optimized pipelines for benchmarking. Enabled by default")
    parser.add_argument("--optimum", action="store_true", help="Use Optimum Intel pipelines for benchmarking")
    parser.add_argument(
        "--lora",
        nargs='*',
        required=False,
        default=None,
        help="Path to LoRA adapters for using OpenVINO GenAI optimized pipelines with LoRA for benchmarking")
    parser.add_argument('--lora_alphas', nargs='*', help='Alphas params for LoRA adapters.', required=False, default=[])
    parser.add_argument("--use_cb", action="store_true", help="Use Continuous Batching inference mode")
    parser.add_argument("--cb_config", required=False, default=None, help="Path to file with Continuous Batching Scheduler settings or dict")
    parser.add_argument("--draft_model", required=False, default=None,
                        help="Path to draft model folder including IR files for Speculative decoding generation")
    parser.add_argument("--draft_device", required=False, default=None, help="Inference device for Speculative decoding of draft model")
    parser.add_argument("--draft_cb_config", required=False, default=None,
                        help="Path to file with Continuous Batching Scheduler settings or dict for Speculative decoding of draft model")
    parser.add_argument("--num_assistant_tokens", required=False, default=None, help="Config option num_assistant_tokens for Speculative decoding")
    parser.add_argument("--assistant_confidence_threshold", required=False, default=None,
                        help="Config option assistant_confidence_threshold for Speculative decoding")
    parser.add_argument(
        '--end_token_stopping',
        action='store_true',
        help='Stop the generation even if output token size does not achieve infer_count or max token size ({DEFAULT_OUTPUT_TOKEN_SIZE}}).'
    )
    parser.add_argument('--set_torch_thread', default=0, type=num_infer_count_type, help='Set the number of Torch thread. ')
    parser.add_argument('-tl', '--tokens_len', type=int, required=False, help='The length of tokens print each time in streaming mode, chunk streaming.')
    parser.add_argument('--streaming', action='store_true', help='Set whether to use streaming mode, only applicable to LLM.')

    return parser.parse_args()


CASE_TO_BENCH = {
    'text_gen': bench_text.run_text_generation_benchmark,
    'image_gen': bench_image.run_image_generation_benchmark,
    'code_gen': bench_text.run_text_generation_benchmark,
    'ldm_super_resolution': bench_ldm_sr.run_ldm_super_resolution_benchmark,
    'speech2text': bench_speech.run_speech_2_txt_benchmark,
}


def main():
    logging_kwargs = {"encoding": "utf-8"} if sys.version_info[1] > 8 else {}
    log.basicConfig(
        format='[ %(levelname)s ] %(message)s',
        level=os.environ.get("LOGLEVEL", log.INFO),
        stream=sys.stdout,
        **logging_kwargs
    )
    args = get_argprser()

    if args.tokens_len is not None and not args.streaming:
        log.error("--tokens_len requires --streaming to be set.")
        exit(1)
    if args.streaming and args.tokens_len is None:
        log.error("--streaming requires --tokens_len to be set.")
        exit(1)
    model_path, framework, model_args, model_name = (
        llm_bench_utils.model_utils.analyze_args(args)
    )
    # Set the device for running OpenVINO backend for torch.compile()
    if model_args['torch_compile_backend']:
        ov_torch_backend_device = str(args.device)
        os.putenv('OPENVINO_TORCH_BACKEND_DEVICE', ov_torch_backend_device.upper())
        os.system('echo [ INFO ] OPENVINO_TORCH_BACKEND_DEVICE=$OPENVINO_TORCH_BACKEND_DEVICE')

    out_str = 'Model path={}'.format(model_path)
    if framework == 'ov':
        out_str += ', openvino runtime version: {}'.format(get_version())
        if model_args['config'].get('PREC_BF16') and model_args['config']['PREC_BF16'] is True:
            log.warning('[Warning] Param bf16/prec_bf16 only work for framework pt. It will be disabled.')
        if 'cpu' in args.device.lower():
            env_omp = os.getenv('OMP_WAIT_POLICY')
            if env_omp is None or env_omp != 'PASSIVE':
                log.warning("It is recommended to set the environment variable OMP_WAIT_POLICY to PASSIVE, "
                            "so that OpenVINO inference can use all CPU resources without waiting.")
            original_torch_thread_nums = torch.get_num_threads()
            if args.set_torch_thread > 0:
                torch.set_num_threads(int(args.set_torch_thread))
            else:
                half_nums_of_torch_threads = original_torch_thread_nums / 2
                if model_args['num_beams'] > 1:
                    torch.set_num_threads(int(half_nums_of_torch_threads))
                else:
                    if half_nums_of_torch_threads > DEFAULT_TORCH_THREAD_NUMS:
                        torch.set_num_threads(DEFAULT_TORCH_THREAD_NUMS)
                    else:
                        torch.set_num_threads(int(half_nums_of_torch_threads))
            log.info(f"The num_beams is {model_args['num_beams']}, update Torch thread num from "
                     f'{original_torch_thread_nums} to {torch.get_num_threads()}, avoid to use the CPU cores for OpenVINO inference.')
    log.info(out_str)
    if args.memory_consumption:
        mem_consumption.start_collect_mem_consumption_thread()
    try:
        if model_args['use_case'] in ['text_gen', 'code_gen']:
            iter_data_list, pretrain_time, iter_timestamp = CASE_TO_BENCH[model_args['use_case']](
                model_path, framework, args.device, args.tokens_len, args.streaming, model_args,
                args.num_iters, mem_consumption)
        else:
            iter_data_list, pretrain_time, iter_timestamp = CASE_TO_BENCH[model_args['use_case']](
                model_path, framework, args.device, model_args, args.num_iters,
                mem_consumption)
        if args.report is not None or args.report_json is not None:
            model_precision = ''
            if framework == 'ov':
                ir_conversion_frontend = llm_bench_utils.model_utils.get_ir_conversion_frontend(model_name, model_path.parts)
                if ir_conversion_frontend != '':
                    framework = framework + '(' + ir_conversion_frontend + ')'
                model_precision = llm_bench_utils.model_utils.get_model_precision(model_path.parts)
            if args.report is not None:
                llm_bench_utils.output_csv.write_result(
                    args.report,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                    iter_timestamp
                )
            if args.report_json is not None:
                llm_bench_utils.output_json.write_result(
                    args.report_json,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                    iter_timestamp
                )
    except Exception:
        log.error('An exception occurred')
        log.info(traceback.format_exc())
        exit(1)
    finally:
        if args.memory_consumption:
            mem_consumption.end_collect_mem_consumption_thread()


if __name__ == '__main__':
    main()
