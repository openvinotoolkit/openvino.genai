import time


class MeanStdPair():
    def __init__(self, mean):
        self.mean = mean


class RawImGenPerfMetrics():
    def __init__(self, unet_inference_durations):
        self.unet_inference_durations = unet_inference_durations


class StableDiffusionHook:
    def __init__(self):
        self.text_encoder_time = 0
        self.unet_time_list = []
        self.vae_decoder_time = 0
        self.text_encoder_step_count = 0
        self.unet_step_count = 0
        self.vae_decoder_step_count = 0
        self.main_model_name = "unet"

    def get_1st_unet_latency(self):
        return self.unet_time_list[0] * 1000 if len(self.unet_time_list) > 0 else 0

    def get_2nd_unet_latency(self):
        return sum(self.unet_time_list[1:]) / (len(self.unet_time_list) - 1) * 1000 if len(self.unet_time_list) > 1 else 0

    def get_first_and_other_unet_infer_duration(self):
        first = self.get_1st_unet_latency()
        other = self.get_2nd_unet_latency()
        return (first, other)

    def get_first_and_other_trans_infer_duration(self):
        return self.get_first_and_other_unet_infer_duration()

    def get_text_encoder_infer_duration(self):
        duration = (self.text_encoder_time / self.text_encoder_step_count) * 1000 if self.text_encoder_step_count > 0 else 0
        return {'text_encoder': duration}

    def get_unet_infer_duration(self):
        mean = (sum(self.unet_time_list) / len(self.unet_time_list)) * 1000 if len(self.unet_time_list) > 0 else 0
        return MeanStdPair(mean=mean)

    def get_vae_decoder_infer_duration(self):
        return (self.vae_decoder_time / self.vae_decoder_step_count) * 1000 if self.vae_decoder_step_count > 0 else 0

    @property
    def raw_metrics(self):
        return RawImGenPerfMetrics(self.unet_time_list)

    def get_text_encoder_step_count(self):
        return self.text_encoder_step_count

    def get_vae_decoder_step_count(self):
        return self.vae_decoder_step_count

    def clear_statistics(self):
        self.text_encoder_time = 0
        self.unet_time_list = []
        self.vae_decoder_time = 0
        self.text_encoder_step_count = 0
        self.unet_step_count = 0
        self.vae_decoder_step_count = 0

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.text_encoder.request

        def my_text_encoder(inputs, share_inputs=True, **kwargs):
            t1 = time.time()
            r = old_text_encoder(inputs, share_inputs=share_inputs, **kwargs)
            t2 = time.time()
            text_encoder_time = t2 - t1
            self.text_encoder_time += text_encoder_time
            self.text_encoder_step_count += 1
            return r
        pipe.text_encoder.request = my_text_encoder

    def new_unet(self, pipe):
        main_model = pipe.unet if pipe.unet is not None else pipe.transformer
        self.main_model_name = "unet" if pipe.unet is not None else "transformer"
        old_unet = main_model.request

        def my_unet(inputs, share_inputs=True, **kwargs):
            t1 = time.time()
            r = old_unet(inputs, share_inputs=share_inputs, **kwargs)
            t2 = time.time()
            unet_time = t2 - t1
            self.unet_time_list.append(unet_time)
            self.unet_step_count += 1
            return r
        main_model.request = my_unet

    def new_vae_decoder(self, pipe):
        old_vae_decoder = pipe.vae_decoder.request

        def my_vae_decoder(inputs, share_inputs=True, **kwargs):
            t1 = time.time()
            r = old_vae_decoder(inputs, share_inputs=share_inputs, **kwargs)
            t2 = time.time()
            vae_decoder_time = t2 - t1
            self.vae_decoder_time += vae_decoder_time
            self.vae_decoder_step_count += 1
            return r
        pipe.vae_decoder.request = my_vae_decoder
