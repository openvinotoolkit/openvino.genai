# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import logging as log
from openvino import get_version
import torch
import traceback
from llm_bench_utils.memory_monitor import MemMonitorWrapper, MemoryDataSummarizer
import llm_bench_utils.output_csv
import llm_bench_utils.output_json
import task.visual_language_generation as bench_vlm
import task.text_generation as bench_text
import task.image_generation as bench_image
import task.super_resolution_generation as bench_ldm_sr
import task.speech_to_text_generation as bench_speech
import task.text_embeddings as bench_text_embed
import task.text_to_speech_generation as bench_text_to_speech
import task.text_reranker as bench_text_rerank
from llm_bench_utils.model_utils import (
    analyze_args,
    get_ir_conversion_frontend,
    get_model_precision
)

DEFAULT_TORCH_THREAD_NUMS = 16


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
    parser.add_argument('-m', '--model', help='model folder including IR files or Pytorch files or path to GGUF model', required=TabError)
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
        help='Enables memory usage information collection mode. If the value is 1, output the maximum memory consumption in warm-up iterations.'
        ' If the value is 2, output the maximum memory consumption in all iterations.\nIt is not recommended to run memory consumption and'
        ' performance benchmarking at the same time. Effect on performance can be reduced by specifying a longer --memory_consumption_delay,'
        ' but the impact is still expected. '
    )
    parser.add_argument(
        "--memory_consumption_delay",
        default=None,
        required=False,
        type=float,
        help="delay for memory consumption check in seconds, smaller value will lead to more precised memory consumption, but may affects performance."
        "It is not recommended to run memory consumption and performance benchmarking in the same time",
    )
    parser.add_argument(
        '-mc_dir',
        '--memory_consumption_dir',
        default=None,
        required=False,
        type=str,
        help='Path to store memory consamption logs and chart.',
    )
    parser.add_argument('-bs', '--batch_size', type=int, default=1, required=False, help='Batch size value')
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
    parser.add_argument("--genai", action="store_true", help="[DEPRECATED] Use OpenVINO GenAI optimized pipelines for benchmarking. Enabled by default")
    parser.add_argument("--optimum", action="store_true", help="Use Optimum Intel pipelines for benchmarking")
    parser.add_argument("--from_onnx", action="store_true", help="Allow initialize Optimum OpenVINO model using ONNX")
    parser.add_argument(
        "--lora",
        nargs='*',
        required=False,
        default=None,
        help="Path to LoRA adapters for using OpenVINO GenAI optimized pipelines with LoRA for benchmarking")
    parser.add_argument('--lora_alphas', nargs='*', help='Alphas params for LoRA adapters.', required=False, default=[])
    parser.add_argument("--lora_mode", choices=["auto", "fuse", "static", "static_rank", "dynamic"], help="LoRA adapters loading mode")
    parser.add_argument("--empty_lora", action="store_true", help="Inference without lora")
    parser.add_argument(
        "--use_cb",
        action="store_true",
        help='Deprecated, will be removed soon! Continues batching mode is used by default. '
        'To switch to SPDA mode, please, set up {"ATTENTION_BACKEND": "SDPA"} in --load_config.'
    )
    parser.add_argument("--cb_config", required=False, default=None, help="Path to file with Continuous Batching Scheduler settings or dict")
    parser.add_argument("--draft_model", required=False, default=None,
                        help="Path to draft model folder including IR files for Speculative decoding generation")
    parser.add_argument("--draft_device", required=False, default=None, help="Inference device for Speculative decoding of draft model")
    parser.add_argument("--draft_cb_config", required=False, default=None,
                        help="Path to file with Continuous Batching Scheduler settings or dict for Speculative decoding of draft model")
    parser.add_argument("--num_assistant_tokens", required=False, default=None,
                        help="Config option num_assistant_tokens for Speculative decoding and Prompt Lookup decoding", type=int)
    parser.add_argument("--assistant_confidence_threshold", required=False, default=None,
                        help="Config option assistant_confidence_threshold for Speculative decoding", type=float)
    parser.add_argument("--max_ngram_size", required=False, default=None,
                        help="Config option assistant_confidence_threshold for Prompt Lookup decoding", type=int)
    parser.add_argument(
        '--end_token_stopping',
        action='store_true',
        help='Stop the generation even if output token size does not achieve infer_count or max token size ({DEFAULT_OUTPUT_TOKEN_SIZE}}).'
    )
    parser.add_argument('--set_torch_thread', default=0, type=num_infer_count_type, help='Set the number of Torch thread. ')
    parser.add_argument('-tl', '--tokens_len', type=int, required=False, help='The length of tokens print each time in streaming mode, chunk streaming.')
    parser.add_argument('--streaming', action='store_true', help='Set whether to use streaming mode, only applicable to LLM.')
    parser.add_argument("--num_steps", type=int, required=False, help="Number of inference steps for image generation")
    parser.add_argument("--height", type=int, required=False, help="Generated image height. Applicable only for Image Generation.")
    parser.add_argument("--width", type=int, required=False, help="Generated image width. Applicable only for Image Generation.")
    parser.add_argument(
        "--static_reshape",
        action="store_true",
        help="Reshape image generation pipeline to specific width & height at pipline creation time. Applicable for Image Generation.")
    parser.add_argument('-mi', '--mask_image', default=None,
                        help='Mask image for Inpainting pipelines. Can be directory or path to single image. Applicable for Image Generation.')
    parser.add_argument('-t', '--task', default=None,
                        choices=['text_gen', 'image_gen', "visual_text_gen", 'speech_to_text', 'image_cls', 'code_gen',
                                 'ldm_super_resolution', 'text_embed', 'text_rerank', 'text_to_speech', "text-to-image", "image-to-image", "inpainting"],
                        help='The task to setup the pipeline type')
    parser.add_argument(
        '--strength', type=float, default=None,
        help='Applicable for Image to image/Inpainting pipelines. Indicates extent to transform the reference `image`. Must be between 0 and 1.')
    parser.add_argument("--disable_prompt_permutation", action="store_true", help="Disable modification prompt from run to run to allow prefix caching")
    parser.add_argument("--embedding_pooling", choices=["cls", "mean", "last_token"], default=None,
                        help="Pooling type CLS or MEAN for encoders, LAST_TOKEN for decoders. "
                             "Different post-processing is applied depending on the padding side. Applicable only for text embeddings")
    parser.add_argument("--embedding_normalize", action="store_true", help="Normalize embeddings. Applicable only for text embeddings")
    parser.add_argument("--embedding_max_length", type=int, default=None,
                        help="Max length for text embeddings. Input text will be padded or truncated to specified value")
    parser.add_argument("--embedding_padding_side", choices=["left", "right"], default=None,
                        help="Side to use for padding 'left' or 'right'. Applicable only for text embeddings")
    parser.add_argument("--reranking_max_length", type=int, default=None,
                        help="Max length for text reranking. Input text will be padded or truncated to specified value")
    parser.add_argument("--reranking_top_n", type=int, default=3,
                        help="Number of top results to return for text reranking")
    parser.add_argument("--texts", nargs='+', default=None,
                        help="List of candidates for reranking based on their relevance to a prompt(query). Applicable for Text Rerank pipeline.")
    parser.add_argument('--texts_file', nargs='+', default=None,
                        help='Texts file(s) in jsonl format with candidates for reranking based on relevance to a prompt(query). '
                        'Multiple files should be separated with space(s). Applicable for Text Rerank pipeline.')
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Apply chat template for LLM. By default chat template is not applied. It's better to use with --disable_prompt_permutation,"
                             " otherwise the prompt will be modified after applying the chat template, so the structure of chat template will not be kept.")
    parser.add_argument("--speaker_embeddings", type=str, default=None,
                        help="Path to .bin or .pt file with speaker embeddings for text to speech scenarios")
    parser.add_argument("--vocoder_path", type=str, default=None,
                        help="Path to vocoder  for text to speech scenarios")
    return parser.parse_args()


CASE_TO_BENCH = {
    'text_gen': bench_text.run_text_generation_benchmark,
    'image_gen': bench_image.run_image_generation_benchmark,
    'code_gen': bench_text.run_text_generation_benchmark,
    'ldm_super_resolution': bench_ldm_sr.run_ldm_super_resolution_benchmark,
    'speech_to_text': bench_speech.run_speech_2_txt_benchmark,
    "visual_text_gen": bench_vlm.run_visual_language_generation_benchmark,
    "text_embed": bench_text_embed.run_text_embddings_benchmark,
    "text_to_speech": bench_text_to_speech.run_text_2_speech_benchmark,
    "text_rerank": bench_text_rerank.run_text_reranker_benchmark
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
    model_path, framework, model_args = (
        analyze_args(args)
    )
    # Set the device for running OpenVINO backend for torch.compile()
    if model_args['torch_compile_backend']:
        ov_torch_backend_device = str(args.device)
        os.putenv('OPENVINO_TORCH_BACKEND_DEVICE', ov_torch_backend_device.upper())
        os.system('echo [ INFO ] OPENVINO_TORCH_BACKEND_DEVICE=$OPENVINO_TORCH_BACKEND_DEVICE')

    out_str = 'Model path={}'.format(model_path)
    if framework == 'ov':
        out_str += ', openvino runtime version: {}'.format(get_version())
        if not model_args['optimum']:
            import openvino_genai
            out_str += ', genai version: {}'.format(openvino_genai.__version__)
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
                        half_nums_of_torch_threads = int(half_nums_of_torch_threads) if int(half_nums_of_torch_threads) else 1
                        torch.set_num_threads(int(half_nums_of_torch_threads))
            log.info(f"The num_beams is {model_args['num_beams']}, update Torch thread num from "
                     f'{original_torch_thread_nums} to {torch.get_num_threads()}, avoid to use the CPU cores for OpenVINO inference.')
    log.info(out_str)
    memory_data_collector = None
    if args.memory_consumption:
        memory_monitor = MemMonitorWrapper()
        if args.memory_consumption_delay:
            memory_monitor.interval = args.memory_consumption_delay
        memory_monitor.create_monitors()
        if args.memory_consumption_dir:
            memory_monitor.set_dir(args.memory_consumption_dir)
        memory_data_collector = MemoryDataSummarizer(memory_monitor)
    try:
        if model_args['use_case'].task in ['text_gen', 'code_gen']:
            iter_data_list, pretrain_time, iter_timestamp = CASE_TO_BENCH[model_args['use_case'].task](
                model_path, framework, args.device, args.tokens_len, args.streaming, model_args,
                args.num_iters, memory_data_collector)
        else:
            iter_data_list, pretrain_time, iter_timestamp = CASE_TO_BENCH[model_args['use_case'].task](
                model_path, framework, args.device, model_args, args.num_iters, memory_data_collector)
        if args.report is not None or args.report_json is not None:
            model_precision = ''
            if framework == 'ov':
                ir_conversion_frontend = get_ir_conversion_frontend(model_args['model_name'], model_path.parts)
                if ir_conversion_frontend != '':
                    framework = framework + '(' + ir_conversion_frontend + ')'
                model_precision = get_model_precision(model_path.parts)
            if args.report is not None:
                llm_bench_utils.output_csv.write_result(
                    args.report,
                    model_args['model_name'],
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                    iter_timestamp,
                    memory_data_collector
                )
            if args.report_json is not None:
                llm_bench_utils.output_json.write_result(
                    args.report_json,
                    model_args['model_name'],
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                    iter_timestamp,
                    memory_data_collector
                )
    except Exception:
        log.error('An exception occurred')
        log.info(traceback.format_exc())
        exit(1)
    finally:
        if memory_data_collector:
            memory_data_collector.memory_monitor.stop()


if __name__ == '__main__':
    main()
