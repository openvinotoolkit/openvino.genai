import time


class WhisperHook:
    def __init__(self):
        self.text_encoder_time = 0
        self.text_decoder_time = 0
        self.text_enc_time_list = []
        self.text_dec_time_list = []
        self.text_encoder_infer_count = 0
        self.text_decoder_infer_count = 0

    def get_text_encoder_latency(self):
        return (self.text_encoder_time / self.text_encoder_infer_count) * 1000 if self.text_encoder_infer_count > 0 else 0
    
    def get_1st_text_enc_latency(self):
        return self.text_enc_time_list[0] * 1000 if len(self.text_enc_time_list) > 0 else 0

    def get_2nd_text_enc_latency(self):
        return sum(self.text_enc_time_list[1:]) / (len(self.text_enc_time_list) - 1) * 1000 if len(self.text_enc_time_list) > 1 else 0

    def get_1st_text_dec_latency(self):
        return self.text_dec_time_list[0] * 1000 if len(self.text_dec_time_list) > 0 else 0

    def get_2nd_text_dec_latency(self):
        return sum(self.text_dec_time_list[1:]) / (len(self.text_dec_time_list) - 1) * 1000 if len(self.text_dec_time_list) > 1 else 0

    def get_text_dec_latency(self):
        return (sum(self.text_dec_time_list) / len(self.text_dec_time_list)) * 1000 if len(self.text_dec_time_list) > 0 else 0

    def get_text_decoder_latency(self):
        return (self.text_decoder_time / self.text_decoder_infer_count) * 1000 if self.text_decoder_infer_count > 0 else 0

    def get_text_encoder_step_count(self):
        return self.text_encoder_infer_count

    def get_text_decoder_step_count(self):
        return self.text_decoder_infer_count

    def clear_statistics(self):
        self.text_encoder_time = 0
        self.text_decoder_time = 0
        self.text_encoder_infer_count = 0
        self.text_decoder_infer_count = 0
        self.text_enc_time_list = []
        self.text_dec_time_list = []

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.model.encoder.request

        def my_text_encoder(inputs, share_inputs=True, share_outputs=True):
            t1 = time.time()
            r = old_text_encoder(inputs, share_inputs, share_outputs)
            t2 = time.time()
            text_encoder_time = t2 - t1
            self.text_enc_time_list.append(text_encoder_time)
            self.text_encoder_time += text_encoder_time
            self.text_encoder_infer_count += 1
            return r
        pipe.model.encoder.request = my_text_encoder

    def new_text_decoder(self, pipe):
        old_text_decoder = pipe.model.forward

        def my_text_decoder(*args, **kwargs):
            t1 = time.time()
            r = old_text_decoder(*args, **kwargs)
            t2 = time.time()
            text_decoder_time = t2 - t1
            self.text_dec_time_list.append(text_decoder_time)
            self.text_decoder_time += text_decoder_time
            self.text_decoder_infer_count += 1
            return r
        pipe.model.forward = my_text_decoder
