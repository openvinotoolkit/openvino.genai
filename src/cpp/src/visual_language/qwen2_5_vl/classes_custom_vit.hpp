// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <dlfcn.h>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen2_5_VL_CustomVIT : public VisionEncoderQwen2_5_VL {
public:
    using VisionEncoderQwen2_5_VL::VisionEncoderQwen2_5_VL;

protected:
    void encode_with_imagepreprocess_cpp(const std::vector<ov::Tensor>& images,
                                         const ov::AnyMap& config_map,
                                         ov::Tensor& out_tensor,
                                         ImageSize& out_rsz_size,
                                         size_t frame_num,
                                         size_t frame_id) override;
};

extern "C" typedef void* pfnCreateQwen2vl(uint32_t llmBatchSize, char* modelPath, uint32_t flag);
extern "C" typedef void pfnReleaseQwen2vl(void*);
extern "C" typedef void pfnInferenceQwen2vl(void* handle, char** inputFiles, char* prompt, float* logits, uint32_t inputFileCount);
extern "C" typedef void pfnInferenceSummaryQwen2vl(void* handle, char** inputFiles, char* prompt, char** outputText, uint32_t* outputLen, uint32_t inputFileCount, uint32_t generationCount);
extern "C" typedef int32_t pfnInferenceVitQwen2vl(void* handle, char** inputFiles, char* prompt, uint8_t** outputLogits, uint32_t** ropeIdxOut, uint32_t* outputLen, uint32_t* ropeIdxLen, uint32_t inputFileCount);

class InputsEmbedderQwen2_5_VL_CustomVIT : public InputsEmbedderQwen2_5_VL {
public:
    InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                       const std::filesystem::path& model_dir,
                                       const std::string& device,
                                       const ov::AnyMap device_config);

    InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap device_config);
    ~InputsEmbedderQwen2_5_VL_CustomVIT();

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

protected:
    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;

private:
#ifdef _MSC_VER
    HMODULE m;
#else
    void* m = nullptr;
#endif
    pfnCreateQwen2vl* create = nullptr;
    pfnReleaseQwen2vl* release = nullptr;
    pfnInferenceVitQwen2vl* inference = nullptr;
    void load_custom_vit_lib();

    void* qwen2vlModel = nullptr;
    uint32_t batchSize = 1;
    char** outputEmbeds = nullptr;
    uint32_t** outputRope = nullptr;

    uint32_t* embedLength = nullptr;
    uint32_t* ropeLength = nullptr;

    char** inputFiles = nullptr;
    char* char_weight_fn = nullptr;
};

} // namespace ov::genai
