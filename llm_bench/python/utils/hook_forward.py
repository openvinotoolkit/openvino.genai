import time


class OVForward:
    def __init__(self):
        self.text_encoder_time = 0
        self.unet_time = 0
        self.vae_decoder_time = 0
        self.text_encoder_infer_nums = 0
        self.unet_infer_nums = 0
        self.vae_decoder_infer_nums = 0

    def get_text_encoder_time(self):
        return self.text_encoder_time

    def get_unet_time(self):
        return self.unet_time

    def get_vae_decoder_time(self):
        return self.vae_decoder_time

    def get_text_encoder_infer_nums(self):
        return self.text_encoder_infer_nums

    def get_unet_infer_nums(self):
        return self.unet_infer_nums

    def get_vae_decoder_infer_nums(self):
        return self.vae_decoder_infer_nums

    def get_unet_vae_avg_time(self):
        avg_time = 0
        if self.unet_infer_nums != 0 and self.vae_decoder_infer_nums != 0:
            avg_time = (self.unet_time + self.vae_decoder_time) / (self.unet_infer_nums + self.vae_decoder_infer_nums)
        return avg_time

    def clear_image_time(self):
        self.text_encoder_time = 0
        self.unet_time = 0
        self.vae_decoder_time = 0
        self.text_encoder_infer_nums = 0
        self.unet_infer_nums = 0
        self.vae_decoder_infer_nums = 0

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.text_encoder.request

        def my_text_encoder(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_text_encoder(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            text_encoder_time = t2 - t1
            self.text_encoder_time += text_encoder_time
            self.text_encoder_infer_nums += 1
            return r
        pipe.text_encoder.request = my_text_encoder

    def new_unet(self, pipe):
        old_unet = pipe.unet.request

        def my_unet(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_unet(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            unet_time = t2 - t1
            self.unet_time += unet_time
            self.unet_infer_nums += 1
            return r
        pipe.unet.request = my_unet

    def new_vae_decoder(self, pipe):
        old_vae_decoder = pipe.vae_decoder.request

        def my_vae_decoder(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_vae_decoder(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            vae_decoder_time = t2 - t1
            self.vae_decoder_time += vae_decoder_time
            self.vae_decoder_infer_nums += 1
            return r
        pipe.vae_decoder.request = my_vae_decoder
