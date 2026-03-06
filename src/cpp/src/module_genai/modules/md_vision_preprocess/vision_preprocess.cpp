// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_preprocess.hpp"

#include "module_genai/modules/models/qwen3_5/vision_preprocess.hpp"
#include "module_genai/modules/models/qwen3_vl/vision_preprocess.hpp"

namespace ov::genai::module {

VisionPreprocess::PTR VisionPreprocess::create(const std::filesystem::path& model_path, VLMModelType model_type) {
	switch (model_type) {
	case VLMModelType::QWEN3_VL:
		return nullptr;
	case VLMModelType::QWEN3_5:
	case VLMModelType::QWEN3_5_MOE:
		return std::make_shared<Qwen3_5VisionPreprocess>(model_path, model_type);
	default:
		return nullptr;
	}
}

}  // namespace ov::genai::module
