# Downloading OpenVINO IRs

Prior to running inference on the model, the OpenVINO IRs formats need to be downloaded. You can do so by substituting the *model_id* and the *model_dir* in the below code snippet to download the Latent Consistency Models:

```
def load_orginal_pytorch_pipeline_componets(skip_models=False, skip_safety_checker=True):
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    feature_extractor = pipe.feature_extractor if not skip_safety_checker else None
    safety_checker = pipe.safety_checker if not skip_safety_checker else None
    text_encoder, unet, vae = None, None, None
    if not skip_models:
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        unet = pipe.unet
        unet.eval()
        vae = pipe.vae
        vae.eval()
    del pipe
    gc.collect()
    return (
        scheduler,
        tokenizer,
        feature_extractor,
        safety_checker,
        text_encoder,
        unet,
        vae,
    )

skip_conversion = (
    TEXT_ENCODER_OV_PATH.exists()
    and UNET_OV_PATH.exists()
    and VAE_DECODER_OV_PATH.exists()
)

(
    scheduler,
    tokenizer,
    feature_extractor,
    safety_checker,
    text_encoder,
    unet,
    vae,
) = load_orginal_pytorch_pipeline_componets(skip_conversion)
```

Run *main.py* to execute the inference pipeline of the model.