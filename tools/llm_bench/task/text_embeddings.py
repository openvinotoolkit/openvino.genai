import os
import time
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.output_csv
import llm_bench_utils.output_json
import llm_bench_utils.output_file
from llm_bench_utils.prompt_utils import get_text_prompt
import llm_bench_utils.gen_output_data as gen_output_data

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}


def run_text_embeddings_optimum(input_text, num, model, tokenizer, args, iter_data_list, prompt_index, bench_hook, proc_id, mem_consumption):
    input_text_list = [input_text] * args['batch_size']
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'padding_side': args.get('emb_padding_side', 'right')}
    max_lenght = args.get('emb_max_length')
    if max_lenght is not None:
        tokenizer_kwargs.update({'padding': 'max_length', 'max_length': max_lenght})
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt', **tokenizer_kwargs)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
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
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    start = time.perf_counter()
    model(**input_data)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    embed_time = end - start
    embed_time_full = end - tok_encode_start
    tm_list = []
    tm_infer_list = []
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        log.debug('latency of all texts:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
    iter_data = gen_output_data.embed_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        total_time=embed_time_full,
        latency=embed_time,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, )
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time, ),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        latency_unit='prompt',
        text_emb=True
    )
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_embeddings_genai(input_text, num, model, tokenizer, args, iter_data_list, prompt_index, bench_hook, proc_id, mem_consumption):
    input_text_list = [input_text] * args['batch_size']
    tokenizer_kwargs = {}
    max_lenght = args.get('emb_max_length')
    if max_lenght is not None:
        tokenizer_kwargs = {'padding': 'max_length', 'max_length': max_lenght}
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt', **tokenizer_kwargs)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
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
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    start = time.perf_counter()
    model.embed_documents(input_text_list)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()
    tm_list = []
    tm_infer_list = []
    embed_time = end - start
    tm_list.append(embed_time)
    tm_infer_list.append(embed_time)
    iter_data = gen_output_data.embed_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_list),
        total_time=embed_time,
        latency=embed_time,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, )
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time, ),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        latency_unit="prompt",
        text_emb=True
    )


def run_text_embddings_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_embeddings_model(model_path, device, mem_consumption, **args)
    iter_data_list = []
    input_text_list = get_text_prompt(args)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, _ in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')

    if not use_genai:
        text_emb_fn = run_text_embeddings_optimum
    else:
        text_emb_fn = run_text_embeddings_genai

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_emb_fn(input_text, num, model, tokenizer, args, iter_data_list, p_idx, bench_hook, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_emb_fn(input_text, num, model, tokenizer, args, iter_data_list,
                            prompt_idx_list[idx], bench_hook, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], False, True, latency_unit="prompt")
    return iter_data_list, pretrain_time, iter_timestamp
