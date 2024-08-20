// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "openpose_detector.hpp"

enum StableDiffusionControlnetPipelinePreprocessMode { JustResize = 0, ResizeAndFill = 1, CropAndResize = 2 };


struct StableDiffusionControlnetPipelineParam {
    std::string prompt;
    std::string negative_prompt;
    std::string input_image;
    std::uint32_t steps;
    std::uint32_t seed;
    float guidance_scale = 7.5;
    float controlnet_conditioning_scale = 1.0;
    int width = 512;
    int height = 512;
    bool use_preprocess_ex = false;
    StableDiffusionControlnetPipelinePreprocessMode mode = JustResize;
};

class StableDiffusionControlnetPipeline {
public:
    StableDiffusionControlnetPipeline(std::string model_path, std::string device);
    ov::Tensor Run(StableDiffusionControlnetPipelineParam&);

private:
    ov::Tensor PreprocessEx(ov::Tensor pose,
                            int result_height,
                            int result_width,
                            StableDiffusionControlnetPipelinePreprocessMode mode,
                            int* pad_width,
                            int* pad_height
    );

    ov::Tensor Preprocess(ov::Tensor pose, int* pad_width, int* pad_height);
    ov::Tensor Postprocess(const ov::Tensor& decoded_image,
                           int pad_height,
                           int pad_width,
                           int result_height,
                           int result_width);
    ov::Tensor TextEncoder(std::string& pos_prompt, std::string& neg_prompt);
    ov::Tensor ControlnetUnet(ov::InferRequest controlnet_infer_request,
                              ov::InferRequest unet_infer_request,
                              ov::Tensor sample,
                              ov::Tensor timestep,
                              ov::Tensor text_embedding_1d,
                              ov::Tensor controlnet_cond,
                              float guidance_scale,
                              float controlnet_conditioning_scale);
    ov::Tensor Unet(ov::InferRequest req,
                    ov::Tensor sample,
                    ov::Tensor timestep,
                    ov::Tensor text_embedding_1d,
                    float guidance_scale);
    ov::Tensor VAE(ov::Tensor sample);

    OpenposeDetector detector;
    ov::CompiledModel text_encoder;
    ov::CompiledModel unet;
    ov::CompiledModel controlnet;
    ov::CompiledModel vae_decoder;
    ov::CompiledModel tokenizer;
};