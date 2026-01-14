// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "module_base.hpp"
#include "logger.hpp"

namespace ov {
namespace genai {

namespace module {

using PipelineModuleInstance = std::vector<IBaseModule::PTR>;

void construct_pipeline(const PipelineModulesDesc& pipeline_modules_desc,
                        PipelineModuleInstance& pipeline_instance,
                        const PipelineDesc::PTR& pipeline_desc);

PipelineModuleInstance sort_pipeline(PipelineModuleInstance& pipeline_instrance);

}  // namespace module
}  // namespace genai
}  // namespace ov