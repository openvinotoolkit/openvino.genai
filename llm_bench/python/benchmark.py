# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import time
from pathlib import Path
import logging as log
import utils.ov_utils
import utils.pt_utils
import utils.model_utils
import torch
import numpy as np
from openvino.runtime import get_version
import PIL
import hashlib
import utils.metrics_print
import utils.output_csv
import traceback
from transformers import set_seed
from PIL import Image
from utils.memory_profile import MemConsumption
from utils.hook_forward import StableDiffusionHook
import utils.output_json
import utils.output_file

FW_UTILS = {'pt': utils.pt_utils, 'ov': utils.ov_utils}

DEFAULT_INFERENCE_STEPS = 20
LCM_DEFAULT_INFERENCE_STEPS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_SUPER_RESOLUTION_STEPS = 50
DEFAULT_SUPER_RESOLUTION_WIDTH = 128
DEFAULT_SUPER_RESOLUTION_HEIGHT = 128
MAX_OUTPUT_TOKEN_SIZE = 64 * 1024

mem_consumption = MemConsumption()
stable_diffusion_hook = StableDiffusionHook()


def gen_iterate_data(
    iter_idx='',
    in_size='',
    infer_count='',
    out_size='',
    gen_time='',
    latency='',
    res_md5='',
    max_rss_mem='',
    max_shared_mem='',
    prompt_idx='',
    tokenization_time=[],
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['output_size'] = out_size
    iter_data['generation_time'] = gen_time
    iter_data['latency'] = latency
    iter_data['result_md5'] = res_md5
    iter_data['first_token_latency'] = ''
    iter_data['other_tokens_avg_latency'] = ''
    iter_data['first_token_infer_latency'] = ''
    iter_data['other_tokens_infer_avg_latency'] = ''
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_shared_mem_consumption'] = max_shared_mem
    iter_data['prompt_idx'] = prompt_idx
    iter_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    iter_data['detokenization_time'] = tokenization_time[1] if len(tokenization_time) > 1 else ''
    return iter_data


def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, warmup_md5, prompt_index, bench_hook, model_precision, proc_id):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt')
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    min_gen_tokens = 0 if args['infer_count'] is None else args['infer_count']
    max_gen_tokens = MAX_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    start = time.perf_counter()
    result = model.generate(**input_data, min_new_tokens=int(min_gen_tokens), max_new_tokens=int(max_gen_tokens), num_beams=args['num_beams'], use_cache=True)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = tokenizer.batch_decode(result)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        if 'sum' not in args['model_name'] and result[bs_idx][:input_token_size].equal(input_tokens[bs_idx]):
            generated_text_len = len(result[bs_idx]) - input_tokens[bs_idx].numel()
        else:
            generated_text_len = len(result[bs_idx])
        num_tokens += generated_text_len
        if generated_text_len > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.md5(result_text.encode(), usedforsecurity=False).hexdigest())
    if num == 0:
        warmup_md5[prompt_index] = result_md5_list
    per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    tm_list = bench_hook.get_time_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tm_infer_list = bench_hook.get_time_infer_list()
    iter_data = gen_iterate_data(
        num,
        input_token_size * args['batch_size'],
        len(tm_infer_list),
        num_tokens,
        generation_time,
        per_token_time,
        result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size']
    )
    if num > 0:
        warmup_md5_list = warmup_md5[prompt_index]
        if result_md5_list != warmup_md5_list:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} is different from warm-up's md5 {warmup_md5_list}")
            utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
    else:
        utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
    bench_hook.clear_time_list()
    bench_hook.clear_time_infer_list()


def run_text_generation_benchmark(model_path, framework, device, args, num_iters):
    model, tokenizer, pretrain_time, bench_hook = FW_UTILS[framework].create_text_gen_model(model_path, device, **args)
    model_precision = utils.model_utils.get_model_precision(model_path.parents._parts)
    iter_data_list = []
    warmup_md5 = {}
    input_text_list = utils.model_utils.get_prompts(args)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'prompt nums: {len(input_text_list)}')

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    prompt_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for prompt_idx, input_text in enumerate(input_text_list):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, warmup_md5, prompt_idx, bench_hook, model_precision, proc_id)
    else:
        for prompt_idx, input_text in enumerate(input_text_list):
            for num in range(num_iters + 1):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, warmup_md5, prompt_idx, bench_hook, model_precision, proc_id)

    utils.metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time


def run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    image_width = image_param.get('width', DEFAULT_IMAGE_WIDTH)
    image_height = image_param.get('height', DEFAULT_IMAGE_HEIGHT)
    nsteps = image_param.get('steps', DEFAULT_INFERENCE_STEPS if 'lcm' not in args["model_name"] else LCM_DEFAULT_INFERENCE_STEPS)
    guidance_scale = image_param.get('guidance_scale', None)
    log.info(
        f"[{'warm-up' if num == 0 else num}] Input params: Batch_size={args['batch_size']}, "
        f'steps={nsteps}, width={image_width}, height={image_height}, guidance_scale={guidance_scale}'
    )
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    additional_args = {}
    if guidance_scale is not None:
        additional_args["guidance_scale"] = guidance_scale
    else:
        if 'lcm-sdxl' in args['model_type']:
            additional_args["guidance_scale"] = 1.0
        if 'turbo' in args['model_name']:
            additional_args["guidance_scale"] = 0.0
    input_text_list = [input_text] * args['batch_size']
    if num == 0 and args["output_dir"] is not None:
        for bs_idx, in_text in enumerate(input_text_list):
            utils.output_file.output_image_input_text(in_text, args, image_id, bs_idx, proc_id)
    start = time.perf_counter()
    res = pipe(input_text_list, num_inference_steps=nsteps, height=image_height, width=image_width, **additional_args).images
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    for bs_idx in range(args['batch_size']):
        rslt_img_fn = utils.output_file.output_gen_image(res[bs_idx], args, image_id, num, bs_idx, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())
    generation_time = end - start
    iter_data = gen_iterate_data(
        iter_idx=num,
        infer_count=nsteps,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        stable_diffusion=stable_diffusion_hook
    )
    utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_benchmark(model_path, framework, device, args, num_iters):
    pipe, pretrain_time = FW_UTILS[framework].create_image_gen_model(model_path, device, **args)
    iter_data_list = []
    input_image_list = utils.model_utils.get_image_param_from_prompt_file(args)
    if len(input_image_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')

    if framework == "ov":
        stable_diffusion_hook.new_text_encoder(pipe)
        stable_diffusion_hook.new_unet(pipe)
        stable_diffusion_hook.new_vae_decoder(pipe)

    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(input_image_list)}')

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    prompt_idx_list = [image_id for image_id, image_param in enumerate(input_image_list)]
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for image_id, image_param in enumerate(input_image_list):
                run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id)
    else:
        for image_id, image_param in enumerate(input_image_list):
            for num in range(num_iters + 1):
                run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id)

    utils.metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], False)
    return iter_data_list, pretrain_time


def run_image_classification(model_path, framework, device, args, num_iters=10):
    model, input_size = FW_UTILS[framework].create_image_classification_model(model_path, device, **args)

    data = torch.rand(input_size)

    test_time = []
    iter_data_list = []
    for num in range(num_iters or 10):
        start = time.perf_counter()
        model(data)
        end = time.perf_counter()
        generation_time = end - start
        test_time.append(generation_time)

        iter_data = gen_iterate_data(iter_idx=num, in_size=input_size, infer_count=num_iters, gen_time=generation_time)
        iter_data_list.append(iter_data)
    log.info(f'Processed {num_iters} images in {np.sum(test_time)}s')
    log.info(f'Average processing time {np.mean(test_time)} s')
    return iter_data_list


def run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, image_id, tm_list, proc_id):
    set_seed(args['seed'])
    nsteps = img.get('steps', DEFAULT_SUPER_RESOLUTION_STEPS)
    resize_image_width = img.get('width', DEFAULT_SUPER_RESOLUTION_WIDTH)
    resize_image_height = img.get('height', DEFAULT_SUPER_RESOLUTION_HEIGHT)
    log.info(
        f"[{'warm-up' if num == 0 else num}] Input params: steps={nsteps}, "
        f'resize_width={resize_image_width}, resize_height={resize_image_height}'
    )
    low_res_img = PIL.Image.open(img['prompt']).convert('RGB')
    low_res_img = low_res_img.resize((resize_image_width, resize_image_height))
    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    res = pipe(low_res_img, num_inference_steps=nsteps, tm_list=tm_list)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    result_md5_list = []
    if framework == 'ov':
        rslt_img_fn = utils.output_file.output_gen_image(res[0], args, image_id, num, None, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())

    generation_time = end - start
    iter_data = gen_iterate_data(
        iter_idx=num,
        infer_count=nsteps,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
    )
    utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn)
    utils.metrics_print.print_ldm_unet_vqvae_infer_latency(num, iter_data, tm_list, warm_up=(num == 0))


def run_ldm_super_resolution_benchmark(model_path, framework, device, args, num_iters):
    pipe, pretrain_time = FW_UTILS[framework].create_ldm_super_resolution_model(model_path, device, **args)
    iter_data_list = []
    tm_list = []
    input_image_list = utils.model_utils.get_image_param_from_prompt_file(args)
    if len(input_image_list) > 0:
        images = []
        for image in input_image_list:
            if args['prompt'] is None and args['prompt_file'] is None:
                raise RuntimeError('==Failure image is empty ==')
            elif args['prompt_file'] is not None:
                image['prompt'] = os.path.join(os.path.dirname(args['prompt_file']), image['prompt'].replace('./', ''))
            image['prompt'] = Path(image['prompt'])
            images.append(image)
    else:
        if args['images'] is not None:
            images = Path(args['images'])
            if images.is_dir():
                images = list(images.glob('*'))
            else:
                images = [images]
        else:
            raise RuntimeError('==Failure image is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(images)}')

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    for num in range(num_iters + 1):
        image_id = 0
        for img in images:
            if num == 0:
                if args["output_dir"] is not None:
                    utils.output_file.output_image_input_text(str(img['prompt']), args, image_id, None, proc_id)
                log.info(f"[{'warm-up' if num == 0 else num}] Input image={img['prompt']}")
            run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, image_id, tm_list, proc_id)
            tm_list.clear()
            image_id = image_id + 1
    utils.metrics_print.print_average(iter_data_list, [], 0, False)

    return iter_data_list, pretrain_time


def num_iters_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError('Minimum input value is 0')
    return x


def num_infer_count_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError('Minimum input value is 1')
    elif x > MAX_OUTPUT_TOKEN_SIZE:
        raise argparse.ArgumentTypeError(f'Max input value is {MAX_OUTPUT_TOKEN_SIZE}')
    return x


def get_argprser():
    parser = argparse.ArgumentParser('LLM benchmarking tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='model folder including IR files or Pytorch files', required=TabError)
    parser.add_argument('-d', '--device', default='cpu', help='inference device')
    parser.add_argument('-r', '--report', help='report csv')
    parser.add_argument('-rj', '--report_json', help='report json')
    parser.add_argument('-f', '--framework', default='ov', help='framework')
    parser.add_argument('-p', '--prompt', default=None, help='one prompt')
    parser.add_argument('-pf', '--prompt_file', default=None, help='prompt file in jsonl format')
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
        default='openvino',
        required=False,
        help='Enables running the torch.compile() with specified backend: pytorch or openvino (default)',
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
    utils.model_utils.add_stateful_model_arguments(parser)

    return parser.parse_args()


CASE_TO_BENCH = {
    'text_gen': run_text_generation_benchmark,
    'image_gen': run_image_generation_benchmark,
    'image_cls': run_image_classification,
    'code_gen': run_text_generation_benchmark,
    'ldm_super_resolution': run_ldm_super_resolution_benchmark,
}


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=os.environ.get("LOGLEVEL", log.INFO), stream=sys.stdout)
    args = get_argprser()
    model_path, framework, model_args, model_name = utils.model_utils.analyze_args(args)

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
    log.info(out_str)
    if args.memory_consumption:
        mem_consumption.start_collect_mem_consumption_thread()
    try:
        iter_data_list, pretrain_time = CASE_TO_BENCH[model_args['use_case']](model_path, framework, args.device, model_args, args.num_iters)
        if args.report is not None or args.report_json is not None:
            model_precision = ''
            if framework == 'ov':
                ir_conversion_frontend = utils.model_utils.get_ir_conversion_frontend(model_name, model_path.parents._parts)
                if ir_conversion_frontend != '':
                    framework = framework + '(' + ir_conversion_frontend + ')'
                model_precision = utils.model_utils.get_model_precision(model_path.parents._parts)
            if args.report is not None:
                utils.output_csv.write_result(
                    args.report,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                )
            if args.report_json is not None:
                utils.output_json.write_result(
                    args.report_json,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
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
