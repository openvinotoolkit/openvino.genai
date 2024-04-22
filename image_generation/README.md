## Image generation

The current folder contains:
- Common folder with:
	- [Diffusers](./common/diffusers) library to simplify writing image generation pipelines
	- [imwrite](./common/imwrite) library to dump `ov::Tensor` to `.bmp` image
- Image generation samples:
	- [Stable Diffuison (with LoRA) C++ image generation pipeline](./stable_diffusion_1_5/cpp)
