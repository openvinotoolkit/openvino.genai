// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>

#include "image_generation/schedulers/ischeduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

#include "json_utils.hpp"
namespace {

void batch_copy(ov::Tensor src, ov::Tensor dst, size_t src_batch, size_t dst_batch, size_t batch_size = 1) {
    const ov::Shape src_shape = src.get_shape(), dst_shape = dst.get_shape();
    ov::Coordinate src_start(src_shape.size(), 0), src_end = src_shape;
    ov::Coordinate dst_start(dst_shape.size(), 0), dst_end = dst_shape;

    src_start[0] = src_batch;
    src_end[0] = src_batch + batch_size;

    dst_start[0] = dst_batch;
    dst_end[0] = dst_batch + batch_size;

    ov::Tensor(src, src_start, src_end).copy_to(ov::Tensor(dst, dst_start, dst_end));
}

ov::Tensor tensor_batch_copy(const ov::Tensor input, const size_t num_images_per_prompt, size_t batch_size_multiplier) {
    ov::Shape repeated_shape = input.get_shape();
    repeated_shape[0] *= num_images_per_prompt;
    ov::Tensor tensor_repeated(input.get_element_type(), repeated_shape);
    for (size_t n = 0; n < num_images_per_prompt; ++n) {
        batch_copy(input, tensor_repeated, 0, n);
    }
    return tensor_repeated;
}

const std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "model_index.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using ov::genai::utils::read_json_param;

    return data["_class_name"].get<std::string>();
}

} // namespace


namespace ov {
namespace genai {

enum class PipelineType {
    TEXT_2_IMAGE = 0,
    IMAGE_2_IMAGE = 1
};

class DiffusionPipeline {
public:
    explicit DiffusionPipeline(PipelineType pipeline_type) :
        m_pipeline_type(pipeline_type) { }

    ImageGenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const ImageGenerationConfig& generation_config) {
        m_generation_config = generation_config;
        m_generation_config.validate();
    }

    void set_scheduler(std::shared_ptr<Scheduler> scheduler) {
        auto casted = std::dynamic_pointer_cast<IScheduler>(scheduler);
        OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
        m_scheduler = casted;
    }

    virtual void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) = 0;

    virtual void compile(const std::string& device, const ov::AnyMap& properties) = 0;

    virtual ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const = 0;

    virtual ov::Tensor generate(const std::string& positive_prompt, ov::Tensor initial_image, const ov::AnyMap& properties) = 0;

    virtual ~DiffusionPipeline() = default;

protected:
    virtual void initialize_generation_config(const std::string& class_name) = 0;

    virtual void check_image_size(const int height, const int width) const = 0;

    virtual void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const = 0;

    PipelineType m_pipeline_type;
    std::shared_ptr<IScheduler> m_scheduler;
    ImageGenerationConfig m_generation_config;
};

} // namespace genai
} // namespace ov
