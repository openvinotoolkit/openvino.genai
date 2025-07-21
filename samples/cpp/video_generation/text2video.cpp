// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <random>
#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/image_generation/image_generation_perf_metrics.hpp"

#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/t5_encoder_model.hpp"
#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "image_generation/stable_diffusion_xl_pipeline.hpp"
#include "image_generation/stable_diffusion_3_pipeline.hpp"
#include "image_generation/flux_pipeline.hpp"

#include "utils.hpp"

// TODO: support video2video, inpainting?
// TODO: decode, perf metrics, set_scheduler, set/get_generation_config, reshape, compile, clone()
// TODO: image->video
// TODO: LoRA?
namespace ov::genai {
struct LTXVideoTransformer3DModel {
    LTXVideoTransformer3DModel(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties) {}
};
struct AutoencoderKLLTXVideo {
    AutoencoderKLLTXVideo(const std::filesystem::path& dir, const std::string& device, const ov::AnyMap& properties) {}
};
struct LTXPipeline {
    std::chrono::steady_clock::duration m_load_time_ms{0};
    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    LTXPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties) {
        // TODO: move to common
        const std::filesystem::path model_index_path = models_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        const std::string class_name = data["_class_name"].get<std::string>();
        OPENVINO_ASSERT(class_name == "LTXPipeline");

        set_scheduler(Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        OPENVINO_ASSERT("T5EncoderModel" == text_encoder);
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties);

        const std::string transformer = data["transformer"][1].get<std::string>();
        OPENVINO_ASSERT("LTXVideoTransformer3DModel" == transformer);
        m_transformer = std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties);

        const std::string vae = data["vae"][1].get<std::string>();
        OPENVINO_ASSERT("AutoencoderKLLTXVideo" == vae);
        m_vae = std::make_shared<AutoencoderKLLTXVideo>(models_dir / "vae_decoder", device, properties);

        initialize_generation_config(class_name);
    }
        /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
            const std::string& positive_prompt,
            Properties&&... properties) {
        return generate(positive_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }
    void initialize_generation_config(const std::string& class_name) {
        // TODO: move to common
    }
    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        // TODO: move to common
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms += stop_time - start_time;
    }
    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        // TODO: move to common
        auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
        OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
        m_scheduler = casted;
    }
};
}  // namespace ov::genai

namespace {
std::unique_ptr<ov::genai::LTXPipeline> create_LTXPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties) {
    // TODO: move to common
    const std::string class_name = get_class_name(models_dir);
    auto start_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT("LTXPipeline" == class_name);
    std::unique_ptr<ov::genai::LTXPipeline> impl = std::make_unique<ov::genai::LTXPipeline>(models_dir, device, properties);
    impl->save_load_time(start_time);
    return impl;
}
}

namespace ov::genai {
struct Text2VideoPipeline {
    std::unique_ptr<LTXPipeline> m_impl;
    Text2VideoPipeline(const std::filesystem::path& models_dir, const std::string& device, const ov::AnyMap& properties = {}) :
        m_impl{create_LTXPipeline(models_dir, device, properties)} {}
    /**
     * Generates image(s) based on prompt and other image generation parameters
     * @param positive_prompt Prompt to generate image(s) from
     * @param negative_prompt
     * @param properties Image generation parameters specified as properties. Values in 'properties' override default value for generation parameters.
     * @returns A tensor which has dimensions [num_images_per_prompt, height, width, 3]
     */
    ov::Tensor generate(const std::string& positive_prompt, const std::string& negative_prompt, const ov::AnyMap& properties = {}) {
        // TODO: explicit negative_prompt arg instead of Property? What other args can be exposed that way?
    }

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ov::Tensor, Properties...> generate(
        const std::string& positive_prompt,
        const std::string& negative_prompt,
        Properties&&... properties
    ) {
        return generate(positive_prompt, negative_prompt, ov::AnyMap{std::forward<Properties>(properties)...});
    }
};
}  // namespace ov::genai

int main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc == 3, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>'");

    const std::string models_dir = argv[1], prompt = argv[2];
    // TODO: Test GPU, NPU, HETERO, MULTI, AUTO
    const std::string device = "CPU";  // GPU can be used as well

    ov::genai::Text2VideoPipeline pipe(models_dir, device);
    // ov::Tensor image = pipe.generate(prompt,
    //     ov::genai::width(512),
    //     ov::genai::height(512),
    //     ov::genai::num_inference_steps(20),
    //     ov::genai::num_images_per_prompt(1),
    //     ov::genai::callback(progress_bar));

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
