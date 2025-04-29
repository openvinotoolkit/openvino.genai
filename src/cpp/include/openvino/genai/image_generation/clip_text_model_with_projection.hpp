// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/clip_text_model.hpp"

namespace ov {
namespace genai {

class CLIPTextModelWithProjection : public CLIPTextModel {
public:
    using CLIPTextModel::CLIPTextModel;
};

} // namespace genai
} // namespace ov
