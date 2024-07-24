# stable diffusion 1.5 controlnet pipeline

## model conversion

### tokenizers

check: https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/

> convert_tokenizer openai/clip-vit-large-patch14 --with-detokenizer -o models/tokenizer

### stable diffusion 1.5 controlnet

```
conda create -n ov_sd_controlnet python==3.11
conda activate ov_sd_controlnet
pip install -r ../../common/detectors/scripts/requirements.txt
pip install -r scripts/requirements.txt
```