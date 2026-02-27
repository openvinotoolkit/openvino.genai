import time
import copy
import logging as log
import llm_bench_utils.hook_greedy_search


logger = log.getLogger(__name__)


class WhisperHook:
    def __init__(self):
        self.enc_infer_count = 0
        self.time_data = []
        self.latency_list = []
        self.tm_list = []
        self.tm_infer_list = []
        self.greedy_hook = None

    def get_time_list(self):
        first_token_latency = 0
        for data in self.time_data:
            if 'enc_token_time' in data:
                first_token_latency += data['enc_token_time']
            if 'dec_token_time' in data:
                first_token_latency += data['dec_token_time'][0]
                self.tm_list.extend(copy.deepcopy(data['dec_token_time'][1:]))
        self.tm_list.insert(0, first_token_latency)
        return self.tm_list

    def get_time_infer_list(self):
        first_infer_latency = 0
        for data in self.time_data:
            if 'enc_infer_time' in data:
                first_infer_latency += data['enc_infer_time']
            if 'dec_infer_time' in data:
                first_infer_latency += data['dec_infer_time'][0]
                self.tm_infer_list.extend(copy.deepcopy(data['dec_infer_time'][1:]))
        self.tm_infer_list.insert(0, first_infer_latency)
        return self.tm_infer_list

    def get_whisper_latency(self):
        self.latency_list.clear()
        for data in self.time_data:
            latency_data = {}
            if 'enc_token_time' and 'enc_infer_time' in data:
                latency_data['enc_token_time'] = round(data['enc_token_time'] * 1000, 2)
                latency_data['enc_infer_time'] = round(data['enc_infer_time'] * 1000, 2)
            if 'dec_token_time' in data:
                dec_token_count = len(data['dec_token_time'])
                dec_infer_count = len(data['dec_infer_time'])
                latency_data['dec_token_count'] = dec_token_count
                latency_data['dec_infer_count'] = dec_infer_count
                latency_data['dec_1st_token_time'] = round(data['dec_token_time'][0] * 1000, 2) if dec_token_count > 0 else 'NA'
                latency_data['dec_2nd_tokens_time'] = round(sum(data['dec_token_time'][1:]) * 1000 / (dec_token_count - 1), 2) if dec_token_count > 1 else 'NA'
                latency_data['dec_1st_infer_time'] = round(data['dec_infer_time'][0] * 1000, 2) if dec_infer_count > 0 else 'NA'
                latency_data['dec_2nd_infers_time'] = round(sum(data['dec_infer_time'][1:]) * 1000 / (dec_infer_count - 1), 2) if dec_infer_count > 1 else 'NA'
            self.latency_list.append(latency_data)

    def print_whisper_latency(self, iter, prompt_idx):
        self.get_whisper_latency()
        str = ''
        for idx, data in enumerate(self.latency_list):
            title = f'[ INFO ] [{iter}][P{prompt_idx}][L{idx}]'
            if 'enc_token_time' and 'enc_infer_time' in data:
                str += \
                    f"{title} encoder token latency: {data['enc_token_time']:.2f} ms/token, " \
                    f"encoder infers latency: {data['enc_infer_time']:.2f} ms/infer"
            if 'dec_1st_token_time' and 'dec_2nd_tokens_time' in data:
                str += \
                    f"\n{title} decoder first token latency: {data['dec_1st_token_time']} ms, " \
                    f"decoder other tokens latency: {data['dec_2nd_tokens_time']} ms/token, " \
                    f"decoder tokens count: {data['dec_token_count']}\n"
            if 'dec_1st_infer_time' and 'dec_2nd_infers_time' in data:
                str += \
                    f"{title} decoder first infer latency: {data['dec_1st_infer_time']} ms, " \
                    f"decoder other infers latency: {data['dec_2nd_infers_time']} ms/infer, " \
                    f"decoder infers count: {data['dec_infer_count']}"
            if idx < len(self.latency_list) - 1:
                str += '\n'
        return str

    def clear_statistics(self):
        self.enc_infer_count = 0
        self.time_data.clear()
        self.tm_list.clear()
        self.tm_infer_list.clear()
        if self.greedy_hook is not None:
            self.greedy_hook.clear_time_list()
            self.greedy_hook.clear_time_infer_list()

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.model.encoder.forward

        def my_text_encoder(*args, **kwargs):
            t1 = time.time()
            r = old_text_encoder(*args, **kwargs)
            t2 = time.time()
            text_encoder_token_time = t2 - t1
            if self.enc_infer_count > 0:
                prev_loop_data = self.time_data[self.enc_infer_count - 1]
                prev_loop_data['enc_token_time'] = text_encoder_token_time
            return r
        pipe.model.encoder.forward = my_text_encoder

    def new_text_encoder_request(self, pipe):
        old_text_encoder_request = pipe.model.encoder.request

        def my_text_encoder_request(*args, **kwargs):
            loop_data = {}
            t1 = time.time()
            r = old_text_encoder_request(*args, **kwargs)
            t2 = time.time()
            text_encoder_infer_time = t2 - t1
            loop_data['enc_infer_time'] = text_encoder_infer_time
            self.time_data.append(loop_data)
            self.enc_infer_count += 1
            return r
        pipe.model.encoder.request = my_text_encoder_request

    def new_text_sample(self, pipe):
        self.greedy_hook = llm_bench_utils.hook_greedy_search.GreedySearchHook()
        self.greedy_hook.new_forward(pipe.model)

    def new_generate(self, pipe):
        old_generate = pipe.model.generate

        def my_generate(**kwargs):
            r = old_generate(**kwargs)
            self.set_decoder_time_data()
            return r
        pipe.model.generate = my_generate

    def set_decoder_time_data(self):
        if self.enc_infer_count > 0:
            prev_loop_data = self.time_data[self.enc_infer_count - 1]
            if self.greedy_hook is not None and (self.greedy_hook.get_time_list() or self.greedy_hook.get_time_infer_list()):
                prev_loop_data['dec_token_time'] = copy.deepcopy(self.greedy_hook.get_time_list())
                prev_loop_data['dec_infer_time'] = copy.deepcopy(self.greedy_hook.get_time_infer_list())
                self.greedy_hook.clear_time_list()
                self.greedy_hook.clear_time_infer_list()
