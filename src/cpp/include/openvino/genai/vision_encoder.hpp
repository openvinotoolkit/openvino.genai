// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/processor_config.hpp"
#include <openvino/openvino.hpp>

namespace ov::genai {
struct HeightWidth {
    size_t height, width;
};

struct EncodedImage {
    ov::Tensor resized_source;
    HeightWidth resized_source_size;
    ov::Tensor slices;
    std::vector<HeightWidth> slices_sizes;
};

class OPENVINO_GENAI_EXPORTS VisionEncoder {
public:
    ov::InferRequest m_encoder;
    ProcessorConfig m_processor_config;

    explicit VisionEncoder(const ov::InferRequest& encoder, const ProcessorConfig& processor_config=ProcessorConfig{}) :
        m_encoder{encoder}, m_processor_config{processor_config} {}

    explicit VisionEncoder(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    EncodedImage encode(const ov::Tensor& image) {
        return encode(image, m_processor_config);
    }

    EncodedImage encode(const ov::Tensor& image, const ProcessorConfig& config);
};
}
