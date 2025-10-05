## [!NOTE]
> Currently llm_bench only supports json files with the suffix .jsonl.
> If there is no prompt file, the default value is used.
> There can be multiple prompts in the prompt file. You can specify the prompt to run by using the option --prompt_index

## 1.Text Generation
Supported parameters that can be set are:
* `prompt` - input prompt text for the text generation
Prompt file example：
{"prompt": "what is openvino?"}
{"prompt": "A chat between a curious user and an artificial intelligence assistant."}

## 2.Stable-diffusion
Supported parameters that can be set are:
* `steps` - inference steps (default 20)
* `width` - resolution width (default 512)
* `height` - resolution height (default 512)
* `guidance_scale` - guidance scale
* `prompt` - input prompt text for the image generation
Prompt file example：
{"steps":"10", "width":"256", "height":"256", "guidance_scale":"1.0", "prompt": "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, 8K HD Resolution, beautifully background"}

## 3.Ldm-super-resolution
Supported parameters that can be set are:
* `steps` - inference steps (default 50)
* `width` - resize image width (default 128)
* `height` - resize image height (default 128)
* `prompt` - image path
Prompt file example：
{"steps": "20", "width": "256", "height": "256", "prompt": "./image_256x256_size/4.png"}

## 4.Whisper
Supported parameters that can be set are:
* `media` - audio file path
* `language` - language of audio (default <|en|>)
* `timestamp` - timestamp for whisper (default true)
Prompt file example：
{"media": "./audio/intel_ad_90s_128kbps.mp3", "language": "<|en|>", "timestamp":false}
{"media": "./audio/intel_ad_120s_128kbps.mp3", "language": "<|en|>", "timestamp":true}

## 5. Visual Language Models
Supported parameters that can be set are:
* `media` - image file path or url
* `prompt`- input text prompt

## 6. Image to Image Generation
Supported parameters that can be set are:
* `steps` - inference steps (default 20)
* `width` - resolution width (default 512)
* `height` - resolution height (default 512)
* `guidance_scale` - guidance scale
* `strength` - strength of noising of the image converted to latent space
* `prompt` - input prompt text for the image generation
* `media` - image file path or url
Prompt file example：
{"steps":"10", "width":"256", "height":"256", "guidance_scale":"0.8", "prompt": "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, 8K HD Resolution, beautifully background", "media": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"}

## 7. Inpainting Image Generation
Supported parameters that can be set are:
* `steps` - inference steps (default 20)
* `width` - resolution width (default 512)
* `height` - resolution height (default 512)
* `guidance_scale` - guidance scale
* `strength` - strength of noising of the image converted to latent space
* `prompt` - input prompt text for the image generation
* `media` - image file path or url
* `mask_image` - mask image file path or url
Prompt file example：
{"steps":"10", "width":"256", "height":"256", "guidance_scale":"0.8", "prompt": "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, 8K HD Resolution, beautifully background", "media": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png", "mask_image": "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"}

## 8. Text To Speech Models
Supported parameters that can be set are:
* `prompt` - input prompt text for the speech generation
Prompt file example：
{"prompt": "what is openvino?"}
{"prompt": "A chat between a curious user and an artificial intelligence assistant."}