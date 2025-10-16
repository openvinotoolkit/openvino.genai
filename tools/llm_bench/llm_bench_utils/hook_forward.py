import time
import types
import statistics


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


class RAGForwardHook:
    def __init__(self):
        self.tm_list = []
        self.tm_infer_list = []

    def clear_time_list(self):
        """Clear the time list."""
        self.tm_list.clear()

    def get_time_list(self):
        """Return the time list."""
        return self.tm_list

    def clear_time_infer_list(self):
        """Clear the infer time list."""
        self.tm_infer_list = []

    def get_time_infer_list(self):
        """Return the infer time list."""
        return self.tm_infer_list

    def new_forward(self_, model):
        model._orig_forward = model.forward

        def new_forward(self, *args, **kwargs):
            t1 = time.time()
            result = self._orig_forward(*args, **kwargs)
            t2 = time.time()
            self_.tm_list.append(t2 - t1)
            return result

        model.forward = types.MethodType(new_forward, model)

        if model.config.model_type != "qwen3":
            if hasattr(model, "request"):
                old_request = model.request

                def new_request(inputs, share_inputs=True, **kwargs):
                    t1 = time.time()
                    r = old_request(inputs, share_inputs=share_inputs, **kwargs)
                    t2 = time.time()
                    self_.tm_infer_list.append(t2 - t1)
                    return r
                model.request = new_request


class TTSHook:
    def __init__(self):
        self.encoder_model_time = 0
        self.postnet_model_time_list = []
        self.vocoder_model_time = 0
        self.new_decoder_model = None

    def clear_statistics(self):
        self.encoder_model_time = 0
        self.postnet_model_time_list = []
        self.vocoder_model_time = 0
        if self.new_decoder_model:
            self.new_decoder_model.decoder_model_time_list = []

    def new_encoder(self, pipe):
        old_encoder = pipe.encoder.request

        def new_encoder_model(*args, **kwargs):
            t1 = time.time()
            r = old_encoder(*args, **kwargs)
            t2 = time.time()
            encoder_time = t2 - t1
            self.encoder_model_time += encoder_time
            return r
        pipe.encoder.request = new_encoder_model

    def new_decoder(self, pipe):
        old_decoder = pipe.decoder.request

        class new_decoder_model():
            def __init__(self, old_decoder):
                self.decoder = old_decoder
                self.decoder_model_time_list = []

            def __call__(self, *args, **kwargs):
                t1 = time.time()
                r = self.decoder(*args, **kwargs)
                t2 = time.time()
                decoder_time = t2 - t1
                self.decoder_model_time_list.append(decoder_time)
                return r

            def reset_state(self):
                self.decoder.reset_state()

        pipe.decoder.request = new_decoder_model(old_decoder)
        self.new_decoder_model = pipe.decoder.request

    def new_postnet(self, pipe):
        old_postnet = pipe.postnet.request

        def new_postnet_model(*args, **kwargs):
            t1 = time.time()
            r = old_postnet(*args, **kwargs)
            t2 = time.time()
            postnet_time = t2 - t1
            self.postnet_model_time_list.append(postnet_time)
            return r
        pipe.postnet.request = new_postnet_model

    def new_vocoder(self, pipe):
        old_vocoder = pipe.vocoder.request

        def new_vocoder_model(*args, **kwargs):
            t1 = time.time()
            r = old_vocoder(*args, **kwargs)
            t2 = time.time()
            vocoder_time = t2 - t1
            self.vocoder_model_time += vocoder_time
            return r
        pipe.vocoder.request = new_vocoder_model

    def print_tts_latency(self, iter_str, prompt_idx):
        decoder_model_time_list = []
        if self.new_decoder_model:
            decoder_model_time_list = self.new_decoder_model.decoder_model_time_list
        self.decoder_model_time_list = self.new_decoder_model.decoder_model_time_list
        info = f'[{iter_str}][P{prompt_idx}] ' \
               f'encoder duration: {self.encoder_model_time * 1000:.4f}ms; ' \
               f'decoder duration: {sum(decoder_model_time_list) * 1000:.4f}ms, iter num: {len(decoder_model_time_list)}, '
        if len(decoder_model_time_list) > 1:
            info += f'1st decoder token latency: {self.decoder_model_time_list[0] * 1000:.4f}ms, ' \
                    f'2nd decoder token latency: {statistics.mean(self.decoder_model_time_list[1:]) * 1000:.4f}ms; '
        elif len(decoder_model_time_list) > 0:
            info += f'decoder latency: {self.decoder_model_time_list[0] * 1000:.4f}ms; '
        info += f'postnet duration: {sum(self.postnet_model_time_list) * 1000:.4f}ms, iter num: {len(self.postnet_model_time_list)}, '
        if len(self.postnet_model_time_list) > 0:
            info += f'postnet latency: {statistics.mean(self.postnet_model_time_list) * 1000:.4f}ms; '
        info += f'vocoder duration: {self.vocoder_model_time * 1000:.4f}ms;'
        return info
