## Diffusers

`Diffusers` is a small C++ static library, which contains scheduling algorithms ported from Hugging Face [Diffusers](https://huggingface.co/docs/diffusers/index) library as well as some functionality to apply LoRA adapters to OpenVINO models.

The library is written to operate with OpenVINO C++ API objects like `ov::Model` and `ov::Tensor` and can be used in deployment scenarios with OpenVINO Runtime on Edge.

### Functionality

The library contains ports of the following scheduling algorithms:
- [LMSDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete)

And can apply LoRA adapters using `InsertLoRA` transformation to inject weights directly to `ov::Model`.
