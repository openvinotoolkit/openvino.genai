// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2_5_vl/classes_custom_vit.hpp"

#include "utils.hpp"

namespace ov::genai {

inline bool init_global_var() {
    auto env = std::getenv("ENABLE_CUSTOM_VIT");
    if (env) {
        if (env == std::string("1")) {
            std::cout << "== ENABLE_CUSTOM_VIT = true" << std::endl;
            return true;
        }
    }
    std::cout << "== ENABLE_CUSTOM_VIT = false" << std::endl;
    return false;
}
bool g_enable_custom_vit = init_global_var();

InputsEmbedderQwen2_5_VL_CustomVIT::InputsEmbedderQwen2_5_VL_CustomVIT(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2_5_VL(vlm_config, model_dir, device, device_config) {}

InputsEmbedderQwen2_5_VL_CustomVIT::InputsEmbedderQwen2_5_VL_CustomVIT(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer, 
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2_5_VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

static ImageSize smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels) {
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height (" + std::to_string(height) + ") and width (" + std::to_string(width) +
                       ") must be greater than factor (" + std::to_string(factor) + ")");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    size_t h_bar = std::round(static_cast<float>(height) / factor) * factor;
    size_t w_bar = std::round(static_cast<float>(width) / factor) * factor;

    if (h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
        h_bar = std::floor(height / beta / factor) * factor;
        w_bar = std::floor(width / beta / factor) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
        h_bar = std::ceil(height * beta / factor) * factor;
        w_bar = std::ceil(width * beta / factor) * factor;
    }

    return ImageSize{h_bar, w_bar};
}

// image vector = 1: means image
// image vector = 2: means video
void VisionEncoderQwen2_5_VL_CustomVIT::encode_with_imagepreprocess_cpp(const std::vector<ov::Tensor>& images,
                                                                        const ov::AnyMap& config_map,
                                                                        ov::Tensor& out_tensor,
                                                                        ImageSize& out_rsz_size,
                                                                        size_t frame_num,
                                                                        size_t frame_id) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    ov::Shape orig_shape = images[0].get_shape();
    ImageSize target_image_size = smart_resize(orig_shape.at(1),
                                               orig_shape.at(2),
                                               config.patch_size * config.merge_size,
                                               config.min_pixels,
                                               config.max_pixels);

    size_t grid_h = target_image_size.height / config.patch_size;
    size_t grid_w = target_image_size.width / config.patch_size;

    out_tensor = ov::Tensor(); // (infer_output.get_element_type(), out_shape);
    out_rsz_size = ImageSize{grid_h, grid_w};
}

std::vector<ov::genai::EncodedImage> InputsEmbedderQwen2_5_VL_CustomVIT::encode_images(
    const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    for (ov::Tensor& image : single_images) {
        std::cout << "== image = " << image.get_shape() << std::endl;
        
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen2_5_VL_CustomVIT::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images, 
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence
) {
    ov::Shape video_fea_shape = ov::Shape({0, 2048});
    ov::Tensor res_video = ov::Tensor(ov::element::f32, video_fea_shape);

    ov::Shape image_fea_shape({1215,2048});
    ov::Tensor res_image(ov::element::f32, image_fea_shape);
    {
        FILE* pf = fopen("dump_embedding.dat", "rb");
        fread(res_image.data(), res_image.get_byte_size(), 1, pf);
        fclose(pf);
    }

    return {res_video, res_image};
}

} // namespace ov::genai
