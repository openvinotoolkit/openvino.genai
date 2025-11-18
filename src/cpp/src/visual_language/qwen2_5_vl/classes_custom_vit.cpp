// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen2_5_vl/classes_custom_vit.hpp"

#include "utils.hpp"

#include <dlfcn.h>
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

inline std::string init_custom_vit_path() {
    auto env = std::getenv("CUSTOM_VIT_PATH");
    if (env) {
        std::cout << "== CUSTOM_VIT_PATH = " << env << std::endl;
        return std::string(env);
    }
    return std::string();
}
static std::string custom_vit_path = init_custom_vit_path();

inline bool file_exists(const std::string& name) {
    std::ifstream file(name);
    return file.good(); 
}

void InputsEmbedderQwen2_5_VL_CustomVIT::load_custom_vit_lib() {
    int32_t err;
#if defined(_MSC_VER)
    m = LoadLibraryA((custom_vit_path + "\\cm.ocl.qwen2vl.lib.dll").c_str());
#else
    m = dlopen((custom_vit_path + std::string("/libcm.ocl.qwen2vl.lib.so")).c_str(), RTLD_LAZY);
    if (!m) {
        fprintf(stderr, "%s\n", dlerror());
        exit(1);
    }
#endif

#if defined(_MSC_VER)
    create = (pfnCreateQwen2vl*)GetProcAddress(m, "createModelQwen2vl");
    release = (pfnReleaseQwen2vl*)GetProcAddress(m, "releaseModelQwen2vl");
    inference = (pfnInferenceVitQwen2vl*)GetProcAddress(m, "inferenceVitQwen2vl");
#else
    create = (pfnCreateQwen2vl*)dlsym(m, "createModelQwen2vl");
    release = (pfnReleaseQwen2vl*)dlsym(m, "releaseModelQwen2vl");
    inference = (pfnInferenceVitQwen2vl*)dlsym(m, "inferenceVitQwen2vl");
#endif

    std::string model_weight_fn = (custom_vit_path + "/weights/qwen2p5.3b.bf16vit.q40llm");
    size_t len = model_weight_fn.length();
    char_weight_fn = new char[len + 1];
    std::strcpy(char_weight_fn, model_weight_fn.c_str());

    uint32_t flag = 1;
    qwen2vlModel = create(batchSize, char_weight_fn, flag);
    if (nullptr == qwen2vlModel) {
        std::cout << "== create custom vit fail." << std::endl;
        exit(0);
    }

    outputEmbeds = (char**)malloc(batchSize * sizeof(char*));
    outputRope = (uint32_t**)malloc(batchSize * sizeof(uint32_t*));

    embedLength = (uint32_t*)malloc(batchSize * sizeof(uint32_t));
    ropeLength = (uint32_t*)malloc(batchSize * sizeof(uint32_t));

    size_t maxEmbedSize = ((1008 / 28) * (1008 / 28) + 800) * 3584 * sizeof(float);
    size_t maxRopeSize = ((1008 / 28) * (1008 / 28) + 800) * 3 * sizeof(uint32_t);
    for (int32_t ii = 0; ii < batchSize; ii++) {
        outputEmbeds[ii] = (char*)malloc(maxEmbedSize);
    }
    for (int32_t ii = 0; ii < batchSize; ii++) {
        outputRope[ii] = (uint32_t*)malloc(maxRopeSize);
    }

    inputFiles = (char**)malloc(batchSize * sizeof(char*));
    memset(inputFiles, 0, batchSize * sizeof(char*));
    std::string img_fn = custom_vit_path + "/input_img.jpg";
    if (!file_exists(img_fn)) {
        std::cout << "Fail, img file does't exit:" << img_fn << std::endl;
        exit(0);
    }

    for (int32_t ii = 0; ii < batchSize; ii++) {
        size_t len = img_fn.length();
        inputFiles[ii] = (char*)malloc(sizeof(char) * (len + 1));
        img_fn.copy(inputFiles[ii], len, 0);
        inputFiles[ii][len] = '\0';
    }
}

InputsEmbedderQwen2_5_VL_CustomVIT::~InputsEmbedderQwen2_5_VL_CustomVIT() {
    free(embedLength);
    free(ropeLength);
    for (int32_t ii = 0; ii < batchSize; ii++) {
      free(outputEmbeds[ii]);
      free(outputRope[ii]);
      free(inputFiles[ii]);
    }
    if (nullptr != m) {
  #if defined(_MSC_VER)
      FreeLibrary(m);
  #else
      dlclose(m);
  #endif
    }
    free(outputEmbeds);
    free(outputRope);
    free(inputFiles);
    free(char_weight_fn);
}

InputsEmbedderQwen2_5_VL_CustomVIT::InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                                                       const std::filesystem::path& model_dir,
                                                                       const std::string& device,
                                                                       const ov::AnyMap device_config)
    : InputsEmbedderQwen2_5_VL(vlm_config, model_dir, device, device_config) {
    load_custom_vit_lib();
}

InputsEmbedderQwen2_5_VL_CustomVIT::InputsEmbedderQwen2_5_VL_CustomVIT(const VLMConfig& vlm_config,
                                                                       const ModelsMap& models_map,
                                                                       const Tokenizer& tokenizer,
                                                                       const std::filesystem::path& config_dir_path,
                                                                       const std::string& device,
                                                                       const ov::AnyMap device_config)
    : InputsEmbedderQwen2_5_VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    load_custom_vit_lib();
}

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

    // (char*)promptIn.c_str()
    size_t remaining = 1;
    inference(qwen2vlModel, inputFiles, nullptr, (uint8_t**)outputEmbeds, outputRope, embedLength, ropeLength, remaining);
    std::memcpy(res_image.data(), outputEmbeds[0], res_image.get_byte_size());

    // {
    //     FILE* pf = fopen("dump_embedding.dat", "rb");
    //     fread(res_image.data(), res_image.get_byte_size(), 1, pf);
    //     fclose(pf);
    // }

    return {res_video, res_image};
}

} // namespace ov::genai
