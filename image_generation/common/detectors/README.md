## Detectors

`Detectors` is a C++ static library, it encapsulates the implementation of the detector from the [controlnet_aux](https://github.com/huggingface/controlnet_aux) library. This library is designed to facilitate the integration of ControlNet support within Stable Diffusion.

The library is written to operate with OpenVINO C++ API objects like `ov::Model` and `ov::Tensor` and can be used in deployment scenarios with OpenVINO Runtime on Edge.

### Functionality

The library contains ports of the following detectors:
- [OpenposeDetector](https://github.com/huggingface/controlnet_aux/blob/6367d57749002a76900a4fc26c06b82b34f495f7/src/controlnet_aux/open_pose/__init__.py#L70C7-L70C23)
