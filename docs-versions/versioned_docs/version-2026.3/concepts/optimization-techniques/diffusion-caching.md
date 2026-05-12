---
sidebar_position: 5
---

# Diffusion Caching (TaylorSeer Lite)

## Overview
Diffusion Caching is an optimization technique that accelerates diffusion transformers by identifying and reusing computational redundancies in the iterative denoising process. The technique exploits the observation that intermediate activations often change minimally between closely spaced timesteps, making it possible to cache and reuse previous computations instead of performing full forward passes at every step.

OpenVINO GenAI implements **TaylorSeer Lite**, a memory-efficient variant of the TaylorSeer algorithm introduced in [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923). This approach achieves significant inference speedups while maintaining visual quality, without requiring model retraining or architectural modifications.

## Conceptual Model
TaylorSeer Lite uses Taylor series approximation to predict transformer outputs during denoising steps, eliminating the need for full forward passes. Instead of caching features from all transformer layers (which would consume substantial memory), TaylorSeer Lite caches only the output of the final linear layer along with its first derivative.

During cached steps, the transformer computation is completely skipped. The output is extrapolated using the Taylor series approximation based on the cached values.

At regular intervals (controlled by `cache_interval`), a full forward pass is executed to refresh the cache and update derivatives, ensuring prediction accuracy.

## Configuration Interface
TaylorSeer Lite is configured through `ov::genai::TaylorSeerCacheConfig` and exposed in `ov::genai::ImageGenerationConfig` and `ov::genai::VideoGenerationConfig`.

### Parameters
* **`cache_interval`** (`size_t`, defaults to `3`) - Controls how often a full forward pass is performed after warm-up. Once warm-up is finished, TaylorSeer performs a full transformer computation every `cache_interval` steps and uses Taylor-series predictions for the intermediate steps, resulting in up to `cache_interval - 1` predicted (cached) denoising steps between two full computations.

* **`disable_cache_before_step`** (`size_t`, defaults to `6`) -  Number of initial denoising steps during which caching is disabled. In practice, the implementation always performs full computations for steps `0..max(disable_cache_before_step, 2) - 1`, ensuring at least two warm-up steps with no caching to stabilize the derivatives before prediction begins.

* **`disable_cache_after_step`** (`int`, defaults to `-2`) - Step index from which caching is disabled (inclusive) to ensure quality in the final denoising stages. Negative values are interpreted relative to the end of the schedule: `num_inference_steps + disable_cache_after_step`.

## Sample Usage (Python)
[samples/python/image_generation/taylorseer_text2image.py](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/image_generation/taylorseer_text2image.py) demonstrates TaylorSeer Lite usage with performance comparison for image generation.

[samples/python/video_generation/taylorseer_text2video.py](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/video_generation/taylorseer_text2video.py) demonstrates TaylorSeer Lite usage with performance comparison for video generation.

Configuration in code:
```python
taylorseer_config = openvino_genai.TaylorSeerCacheConfig()
taylorseer_config.cache_interval = 5
taylorseer_config.disable_cache_before_step = 2
taylorseer_config.disable_cache_after_step = -1
```

### Image Generation (Flux / StableDiffusion3)
```python
pipe = openvino_genai.Text2ImagePipeline(models_path, device)
# Apply TaylorSeerCacheConfig to generation config
generation_config = pipe.get_generation_config()
generation_config.taylorseer_config = taylorseer_config
pipe.set_generation_config(generation_config)

res = pipe.generate(prompt, num_inference_steps=28)
```

### Video Generation (LTX-Video)
TaylorSeer caching is **enabled by default** for the LTX-Video pipeline. No configuration is required to benefit from it.

```python
pipe = openvino_genai.Text2VideoPipeline(models_path, device)
# TaylorSeer is active out of the box
result = pipe.generate(prompt, num_inference_steps=25)
```

To customize caching parameters, use `set_generation_config()`:
```python
generation_config = pipe.get_generation_config()
generation_config.taylorseer_config = taylorseer_config
pipe.set_generation_config(generation_config)
```

To disable caching entirely:
```python
generation_config = pipe.get_generation_config()
generation_config.taylorseer_config = None  # disable caching
pipe.set_generation_config(generation_config)
```

## Benefits
* By skipping full transformer computations for multiple consecutive steps, inference time is significantly reduced.
* Only the final layer output and derivative are cached, avoiding the memory explosion that would occur with full-layer caching.
* The Taylor approximation maintains high visual similarity to full computation results.
* Speedup scales with transformer computation intensity, input resolution and number of inference steps.

## Current Limitations
* TaylorSeer Lite currently supports Flux and StableDiffusion3 Text2Image pipelines, and LTX-Video Text2Video pipeline.
