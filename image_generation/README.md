## Image Generation

The current folder contains:
- Common folder with:
	- [Diffusers](./common/diffusers) library to simplify writing image generation pipelines
	- [imwrite](./common/imwrite) library to dump `ov::Tensor` to `.bmp` image
- Image generation samples:
	- [Stable Diffuison (with LoRA) C++ image generation pipeline](./stable_diffusion_1_5/cpp)
	- [OpenVINO Latent Consistency Model C++ image generation pipeline](./lcm_dreamshaper_v7/cpp)
