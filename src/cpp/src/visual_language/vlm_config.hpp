// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>

#include <openvino/runtime/properties.hpp>

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

enum class VLMModelType {
    MINICPM,
    LLAVA,
    NANOLLAVA,
    LLAVA_NEXT,
    LLAVA_NEXT_VIDEO,
    INTERNVL_CHAT,
    PHI3_V,
    PHI4MM,
    QWEN2_VL,
    QWEN2_5_VL,
    QWEN3_VL,
    GEMMA3,
    GEMMA4,
    VIDEOCHAT_FLASH_QWEN,
    QWEN3_OMNI,
};

/// @brief A Configuration class passed to VLMPipeline and used to
/// change VLMPipeline's behavior. Corresponds to config.json.
class VLMConfig {
public:
    /// @brief A enum denoting model type.
    VLMModelType model_type;
    /// @brief A size of a single embedding returned by a resampler.
    /// Used to initialize positional embeddings for resampler input.
    size_t hidden_size = 3584;
    /// @brief Multiply embeddings by this value.
    float scale_emb = 1.0f;
    /// @brief A number of embedding vectors representing an image
    /// slice.
    size_t query_num = 64;
    /// @brief A string token denoting start of image embeddings for an
    /// LLM.
    std::string im_start = "<image>";
    /// @brief A string token denoting end of image embeddings for an
    /// LLM.
    std::string im_end = "</image>";
    /// @brief A string token denoting start of image slices row
    /// embeddings for an LLM.
    std::string slice_start = "<slice>";
    /// @brief A string token denoting end of image slices row
    /// embeddings for LLM.
    std::string slice_end = "</slice>";
    /// @brief Start each image (not a slice) with
    /// <image_id>i</image_id>. i is a number.
    bool use_image_id = true;
    /// @brief A string token denoting start of image number region.
    std::string im_id_start = "<image_id>";
    /// @brief A string token denoting end of image number region.
    std::string im_id_end = "</image_id>";
    /// @brief A placeholder for image embeddings in text.
    std::string unk = "<unk>";

    // llava_next specific config params
    std::vector<float> image_newline;
    size_t vision_config_patch_size = 14;

    /// @brief A string token denoting start of image embeddings for InternVL2 model.
    std::string image_start_token = "<img>";
    /// @brief A placeholder for image embeddings in text for InternVL2 model.
    std::string image_context_token = "<IMG_CONTEXT>";
    /// @brief A string token denoting end of image embeddings for InternVL2 model.
    std::string image_end_token = "</img>";
    /// @brief phi3_v and phi4mm new line token embedding to separate images.
    std::vector<float> sub_GN = std::vector(4096, 0.0f);
    std::vector<float> glb_GN = std::vector(4096, 0.0f);

    /// @brief A string token denoting start of vision embeddings for Qwen2VL model.
    std::string vision_start_token = "<|vision_start|>";
    /// @brief A placeholder for image embeddings in text for Qwen2VL model.
    std::string image_pad_token = "<|image_pad|>";
    std::string video_pad_token = "<|video_pad|>";
    /// @brief A string token denoting end of vision embeddings for Qwen2VL model.
    std::string vision_end_token = "<|vision_end|>";

    /// @brief A size of a window for Qwen2.5VL model, used in window attention.
    size_t vision_config_window_size = 112;

    /// @brief A token id per second for Qwen2.5VL model, used in calc position_ids.
    size_t vision_config_tokens_per_second = 2;

    /// @brief A string token denoting start of vision embeddings for gemma3-4b-it model.
    std::string start_of_image = "<start_of_image>";
    /// @brief A placeholder for image embeddings in text for gemma3-4b-it model.
    std::string image_soft_token = "<image_soft_token>";
    /// @brief A string token denoting end of vision embeddings for gemma3-4b-it model.
    std::string end_of_image = "<end_of_image>";

    /// @brief A string token denoting start of image embeddings for Gemma4 model.
    std::string boi_token = "<|image>";
    /// @brief A placeholder for image embeddings in text for Gemma4 model.
    std::string image_token = "<|image|>";
    /// @brief A string token denoting end of image embeddings for Gemma4 model.
    std::string eoi_token = "<image|>";

    /// @brief A string token denoting start of video embeddings
    std::string video_start = "<video>";

    // Qwen3-VL specific config
    /// @brief Number of position embeddings in vision encoder for Qwen3-VL model.
    size_t vision_config_num_position_embeddings = 2304;
    /// @brief DeepStack visual indexes for Qwen3-VL model.
    std::vector<size_t> vision_config_deepstack_visual_indexes;

    // Qwen3-Omni specific config
    /// @brief Whether audio output (speech) is enabled.
    bool enable_audio_output = false;
    /// @brief Audio encoder: number of mel spectrogram bins.
    size_t audio_config_num_mel_bins = 128;
    /// @brief Audio encoder: window size for training.
    size_t audio_config_n_window = 50;
    /// @brief Audio encoder: window size for inference.
    size_t audio_config_n_window_infer = 200;
    /// @brief Talker: number of codec quantizer groups.
    size_t talker_num_code_groups = 16;
    /// @brief Talker: thinker hidden size for projection.
    size_t talker_thinker_hidden_size = 2560;
    // Codec special token IDs
    int64_t talker_codec_bos_id = 2149;
    int64_t talker_codec_eos_token_id = 2150;
    int64_t talker_codec_pad_id = 2148;
    int64_t talker_codec_nothink_id = 2155;
    int64_t talker_codec_think_bos_id = 2156;
    int64_t talker_codec_think_eos_id = 2157;
    // TTS special token IDs (in thinker vocabulary)
    int64_t tts_bos_token_id = -1;
    int64_t tts_eos_token_id = -1;
    int64_t tts_pad_token_id = -1;
    // ChatML role token IDs for talker input construction
    int64_t im_start_token_id = -1;
    int64_t im_end_token_id = -1;
    int64_t system_token_id = -1;
    int64_t user_token_id = -1;
    int64_t assistant_token_id = -1;
    int64_t audio_token_id = -1;
    int64_t image_token_id = -1;
    int64_t video_token_id = -1;
    // Speaker name-to-codec-token mapping
    std::map<std::string, int64_t> speaker_ids;

    /// @brief Default constructor.
    VLMConfig() = default;
    /// @brief Construct VLMConfig from values in json_path.
    /// Keys in the file must match the VLMConfig's members.
    /// @param json_path A path to a file to extract the values from.
    explicit VLMConfig(const std::filesystem::path& config_path);
    /// @brief Default copy constructor.
    /// @param A config to copy from.
    VLMConfig(const VLMConfig&) = default;
};
}  // namespace ov::genai
