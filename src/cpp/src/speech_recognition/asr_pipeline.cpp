// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "openvino/genai/speech_recognition/asr_pipeline.hpp"

#include "speech_recognition/paraformer_impl.hpp"

namespace ov {
namespace genai {

ASRPipeline::ASRPipeline(const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap& properties) {

}

ASRPipeline::ASRPipeline(const std::shared_ptr<ASRPipelineImplBase>& impl)
    : m_impl(impl) {
    assert(m_impl != nullptr);
}

ASRPipeline ASRPipeline::paraformerASR(const std::filesystem::path& model_dir,
                                       const std::string& device,
                                       const ov::AnyMap& properties) {
    auto impl = std::make_shared<ParaformerImpl>(model_dir, device, properties);
    return ASRPipeline(impl);
}

}  // namespace genai
}  // namespace ov