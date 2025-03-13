import json


def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, iter_timestamp):
    metadata = {'model': model, 'framework': framework, 'device': device, 'precision': model_precision,
                'num_beams': model_args['num_beams'], 'batch_size': model_args['batch_size']}
    result = []
    total_iters = len(iter_data_list)
    for i in range(total_iters):
        iter_data = iter_data_list[i]
        generation_time = iter_data['generation_time']
        latency = iter_data['latency']
        first_latency = iter_data['first_token_latency']
        other_latency = iter_data['other_tokens_avg_latency']
        first_token_infer_latency = iter_data['first_token_infer_latency']
        other_token_infer_latency = iter_data['other_tokens_infer_avg_latency']
        rss_mem = iter_data['max_rss_mem_consumption']
        max_sys_mem = iter_data['max_sys_mem_consumption']
        tokenization_time = iter_data['tokenization_time']
        detokenization_time = iter_data['detokenization_time']

        result_md5 = []
        for idx_md5 in range(len(iter_data['result_md5'])):
            result_md5.append(iter_data['result_md5'][idx_md5])

        timestamp_start, timestamp_end = get_timestamp(iter_data['iteration'], iter_data['prompt_idx'], iter_timestamp)

        res_data = {
            'iteration': iter_data['iteration'],
            'input_size': iter_data['input_size'],
            'infer_count': iter_data['infer_count'],
            'generation_time': round(generation_time, 5) if generation_time != '' else generation_time,
            'output_size': iter_data['output_size'],
            'latency': round(latency, 5) if latency != '' else latency,
            'result_md5': result_md5,
            'first_latency': round(first_latency, 5) if first_latency != '' else first_latency,
            'second_avg_latency': round(other_latency, 5) if other_latency != '' else other_latency,
            'first_infer_latency': round(first_token_infer_latency, 5) if first_token_infer_latency != '' else first_token_infer_latency,
            'second_infer_avg_latency': round(other_token_infer_latency, 5) if other_token_infer_latency != '' else other_token_infer_latency,
            'max_rss_mem': round(rss_mem, 5) if rss_mem != '' else -1,
            'max_sys_mem': round(max_sys_mem, 5) if max_sys_mem != '' else -1,
            'prompt_idx': iter_data['prompt_idx'],
            'tokenization_time': round(tokenization_time, 5) if tokenization_time != '' else tokenization_time,
            'detokenization_time': round(detokenization_time, 5) if detokenization_time != '' else detokenization_time,
            'start': timestamp_start,
            'end': timestamp_end
        }

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
        values = [x[key] for x in result if x[key] != '']
        if len(values) > 0:
            results_averaged[key] = round(sum(values) / len(values), 5)

    output_result = {'metadata': metadata, "perfdata": {'compile_time': pretrain_time, 'results': result}}

    if len(results_averaged) > 0:
        output_result['perfdata']['results_averaged'] = results_averaged

    with open(report_file, 'w') as outfile:
        json.dump(output_result, outfile, indent=4)


def get_timestamp(iter_idx, prompt_idx, iter_timestamp):
    timestamp_start = ''
    timestamp_end = ''
    if iter_idx in iter_timestamp.keys():
        if prompt_idx in iter_timestamp[iter_idx].keys():
            timestamp_start = iter_timestamp[iter_idx][prompt_idx]['start']
            timestamp_end = iter_timestamp[iter_idx][prompt_idx]['end']

    return timestamp_start, timestamp_end
