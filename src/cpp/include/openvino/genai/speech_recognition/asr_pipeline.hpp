// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "utils.hpp"

namespace ov {
namespace genai {

// forward declaration
class ASRPipelineImplBase;

class OPENVINO_GENAI_EXPORTS ASRPipeline {
public:
    explicit ASRPipeline(const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap& properties = {});

    static ASRPipeline paraformerASR(const std::filesystem::path& model_dir,
                                     const std::string& device,
                                     const ov::AnyMap& properties = {});

private:
    std::shared_ptr<ASRPipelineImplBase> m_impl;

    explicit ASRPipeline(const std::shared_ptr<ASRPipelineImplBase>& impl);
};

} // namespace genai
} // namespace ov
