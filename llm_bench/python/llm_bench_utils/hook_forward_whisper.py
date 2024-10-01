import time
import copy
import llm_bench_utils.hook_greedy_search


class WhisperHook:
    def __init__(self):
        self.enc_infer_count = 0
        self.time_data = []
        self.greedy_hook = None

    def get_time_list(self):
        """return first loop token time
        """
        time_list = []
        if len(self.time_data) > 0:
            time_list = copy.deepcopy(self.time_data[0]['dec_token_time'])
            time_list.insert(0, self.time_data[0]['enc_infer_time'])
        return time_list

    def get_time_infer_list(self):
        """return first loop infer time
        """
        time_infer_list = []
        if len(self.time_data) > 0:
            time_infer_list = copy.deepcopy(self.time_data[0]['dec_infer_time'])
            time_infer_list.insert(0, self.time_data[0]['enc_infer_time'])
        return time_infer_list
    
    def get_whisper_latency(self, iter, prompt_idx):
        str = ''
        for idx, data in enumerate(self.time_data):
            enc_infer_time = data['enc_infer_time'] * 1000
            dec_token_count = len(data['dec_token_time'])
            dec_infer_count = len(data['dec_infer_time'])
            dec_token_time = sum(data['dec_token_time']) / dec_token_count * 1000 if dec_token_count > 1 else 0
            dec_infer_time = sum(data['dec_infer_time']) / dec_infer_count * 1000 if dec_infer_count > 1 else 0
            str += f"[{iter}][P{prompt_idx}][L{idx}] encoder token latency: {enc_infer_time:.2f} ms/token, " \
                f"decoder tokens latency: {dec_token_time:.2f} ms/token, " \
                f"decoder infers latency: {dec_infer_time:.2f} ms/infer, " \
                f"decoder tokens count: {dec_token_count}, " \
                f"decoder infers count: {dec_infer_count}"
            if idx < len(self.time_data) - 1:
                str += '\n'
        return str

    def clear_statistics(self):
        self.enc_infer_count = 0
        self.time_data.clear()
        self.greedy_hook.clear_time_list()
        self.greedy_hook.clear_time_infer_list()

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.model.encoder.forward

        def my_text_encoder(*args, **kwargs):
            loop_data = {}
            t1 = time.time()            
            r = old_text_encoder(*args, **kwargs)
            t2 = time.time()
            text_encoder_time = t2 - t1
            loop_data['enc_infer_time'] = text_encoder_time
            self.time_data.append(loop_data)
            self.enc_infer_count += 1
            return r
        pipe.model.encoder.forward = my_text_encoder

    def new_text_sample(self, pipe):
        self.greedy_hook = llm_bench_utils.hook_greedy_search.GreedySearchHook()
        self.greedy_hook.new_forward(pipe.model)

    def new_generate(self, pipe):
        old_generate = pipe.model.generate
        def my_generate(attention_mask, **kwargs):
            r = old_generate(attention_mask, **kwargs)
            self.set_decoder_time_data()
            return r
        pipe.model.generate = my_generate

    def set_decoder_time_data(self):
        if self.enc_infer_count > 0:
            prev_data = self.time_data[self.enc_infer_count - 1]
            prev_data['dec_token_time'] = copy.deepcopy(self.greedy_hook.get_time_list())
            prev_data['dec_infer_time'] = copy.deepcopy(self.greedy_hook.get_time_infer_list())
            self.greedy_hook.clear_time_list()
            self.greedy_hook.clear_time_infer_list()