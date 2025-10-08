// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "paraformer_impl.hpp"

namespace ov {
namespace genai {

ParaformerImpl::ParaformerImpl(const std::filesystem::path& model_dir,
                               const std::string& device,
                               const ov::AnyMap& properties)
: ASRPipelineImplBase(model_dir, device, properties)
{

}

ParaformerImpl::~ParaformerImpl()
{}

}  // namespace genai
}  // namespace ov