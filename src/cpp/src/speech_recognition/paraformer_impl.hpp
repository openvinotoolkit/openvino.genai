// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "asr_pipelinee_impl_base.hpp"

namespace ov {
namespace genai {

class ParaformerImpl : public ASRPipelineImplBase {
public:
    explicit ParaformerImpl(const std::filesystem::path& model_dir,
                            const std::string& device,
                            const ov::AnyMap& properties = {});
    virtual ~ParaformerImpl() override;
};

} // namespace genai
} // namespace ov