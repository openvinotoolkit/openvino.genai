# Image generation
## [!NOTE]
> Currently llm_bench only supports json files with the suffix .jsonl.
> If there is no prompt file, the default value is used.

## 1.Stable-diffusion
Supported parameters that can be set are:
* `steps` - inference steps (default 20)
* `width` - resolution width (default 512)
* `height` - resolution height (default 512)
* `guidance_scale` - guidance scale
* `prompt` - input prompt text for the image generation
Prompt file example：
{"steps":"10", "width":"256", "height":"256", "guidance_scale":"1.0", "prompt": "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, 8K HD Resolution, beautifully background"}

## 2.Ldm-super-resolution
Supported parameters that can be set are:
* `steps` - inference steps (default 50)
* `width` - resize image width (default 128)
* `height` - resize image height (default 128)
* `prompt` - Image path
Prompt file example：
{"steps": "20", "width": "256", "height": "256", "prompt": "./image_256x256_size/4.png"}