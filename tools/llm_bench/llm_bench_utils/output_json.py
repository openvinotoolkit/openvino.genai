import json


def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, json_data_list):
    metadata = {'model': model, 'framework': framework, 'device': device, 'precision': model_precision,
                'num_beams': model_args['num_beams'], 'batch_size': model_args['batch_size']}
    result = []
    result_list = json_data_list if len(json_data_list) > 0 else iter_data_list
    total_iters = len(result_list)
    for i in range(total_iters):
        json_data = result_list[i]
        generation_time = json_data['generation_time']
        latency = json_data['latency']
        rss_mem = json_data['max_rss_mem_consumption']
        uss_mem = json_data['max_uss_mem_consumption']
        shared_mem = json_data['max_shared_mem_consumption']
        tokenization_time = json_data['tokenization_time']
        detokenization_time = json_data['detokenization_time']

        result_md5 = []
        for idx_md5 in range(len(json_data['result_md5'])):
            result_md5.append(json_data['result_md5'][idx_md5])

        res_data = {
            'iteration': json_data['iteration'],
            'prompt_idx': json_data['prompt_idx'],
            'input_size': json_data['input_size'],
            'infer_count': json_data['infer_count'],
            'generation_time': round(generation_time, 5) if generation_time != '' else generation_time,
            'output_size': json_data['output_size'],
            'latency': round(latency, 5) if latency != '' else latency,
            'result_md5': result_md5,
            'max_rss_mem': round(rss_mem, 5) if rss_mem != '' else -1,
            'max_uss_mem': round(uss_mem, 5) if uss_mem != '' else -1,
            'max_shared_mem': round(shared_mem, 5) if shared_mem != '' else -1,
            'tokenization_time': round(tokenization_time, 5) if tokenization_time != '' else tokenization_time,
            'detokenization_time': round(detokenization_time, 5) if detokenization_time != '' else detokenization_time,
        }

        if 'loop' not in json_data:
            first_latency = json_data['first_token_latency']
            other_latency = json_data['other_tokens_avg_latency']
            first_token_infer_latency = json_data['first_token_infer_latency']
            other_token_infer_latency = json_data['other_tokens_infer_avg_latency']
            res_data['first_latency'] = round(first_latency, 5) if first_latency != '' else first_latency
            res_data['second_avg_latency'] = round(other_latency, 5) if other_latency != '' else other_latency
            res_data['first_infer_latency'] = round(first_token_infer_latency, 5) if first_token_infer_latency != '' else first_token_infer_latency
            res_data['second_infer_avg_latency'] = round(other_token_infer_latency, 5) if other_token_infer_latency != '' else other_token_infer_latency
        else:
            res_data['loop'] = json_data['loop']
        result.append(res_data)

    output_result = {'metadata': metadata, "perfdata": {'compile_time': pretrain_time, 'results': result}}

    with open(report_file, 'w') as outfile:
        json.dump(output_result, outfile)
