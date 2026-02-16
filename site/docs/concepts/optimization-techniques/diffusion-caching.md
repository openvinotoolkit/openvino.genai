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
TaylorSeer Lite is configured through `ov::genai::TaylorSeerCacheConfig` and exposed in `ov::genai::ImageGenerationConfig`.

### Parameters
* **`cache_interval`** (`size_t`, defaults to `3`) - Compute interval between full forward passes. After a full computation, the cached output is reused for this many subsequent denoising steps before refreshing with a new full forward pass.

* **`disable_cache_before_step`** (`size_t`, defaults to `6`) - Number of initial steps where caching is disabled. Full computation is performed for steps 0 through `disable_cache_before_step - 1` to establish stable derivatives before prediction begins. This warmup period improves prediction accuracy.

* **`disable_cache_after_step`** (`int`, defaults to `-2`) - Step index after which caching is disabled to ensure quality in final denoising stages. Negative values are interpreted relative to the end: `num_inference_steps + disable_cache_after_step`.

## Sample Usage (Python)
[samples/python/image_generation/taylorseer_text2image.py](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/image_generation/taylorseer_text2image.py) demonstrates TaylorSeer Lite usage with performance comparison.

Basic usage:
```bash
python taylorseer_text2image.py \
    ./flux.1-dev/FP16 \
    "a beautiful sunset over mountains" \
    --steps 28 \
    --cache-interval 3 \
    --disable-before 4 \
    --disable-after -2
```

Configuration in code:
```python
taylorseer_config = openvino_genai.TaylorSeerCacheConfig(
    cache_interval=3,
    disable_cache_before_step=4,
    disable_cache_after_step=-2,
)
```

Pipeline creation and generation:
```python
pipe = openvino_genai.Text2ImagePipeline(models_path, device)
# Apply TaylorSeerCacheConfig to generation config
generation_config = pipe.get_generation_config()
generation_config.taylorseer_config = taylorseer_config
pipe.set_generation_config(generation_config)

res = pipe.generate(prompt, num_inference_steps=28)
```

## Benefits
* By skipping full transformer computations for multiple consecutive steps, inference time is significantly reduced.
* Only the final layer output and derivative are cached, avoiding the memory explosion that would occur with full-layer caching.
* The Taylor approximation maintains high visual similarity to full computation results.
* Speedup scales with transformer computation intensity, input resolution and number of inference steps.

## Current Limitations
* The current implementation supports Flux models only; support for other models will be added in subsequent releases.
