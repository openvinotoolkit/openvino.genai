import time


class OVForward:
    def __init__(self):
        self.tm_list = []

    def clear_tm_list(self):
        self.tm_list.clear()

    def get_tm_list(self):
        return self.tm_list

    def new_text_encoder(self, pipe):
        old_text_encoder = pipe.text_encoder.request

        def my_text_encoder(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_text_encoder(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            self.tm_list.append(t2 - t1)
            return r
        pipe.text_encoder.request = my_text_encoder

    def new_unet(self, pipe):
        old_unet = pipe.unet.request

        def my_unet(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_unet(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            self.tm_list.append(t2 - t1)
            return r
        pipe.unet.request = my_unet

    def new_vae_decoder(self, pipe):
        old_vae_decoder = pipe.vae_decoder.request

        def my_vae_decoder(inputs, shared_memory=True, **kwargs):
            t1 = time.time()
            r = old_vae_decoder(inputs, shared_memory=shared_memory, **kwargs)
            t2 = time.time()
            self.tm_list.append(t2 - t1)
            return r
        pipe.vae_decoder.request = my_vae_decoder
