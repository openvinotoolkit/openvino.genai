// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/image_generation/clip_text_model.hpp"

namespace ov {
namespace genai {

class CLIPTextModelWithProjection : public CLIPTextModel {
public:
    using CLIPTextModel::CLIPTextModel;

    std::shared_ptr<CLIPTextModel> clone() {
        std::shared_ptr<CLIPTextModelWithProjection> cloned = std::make_shared<CLIPTextModelWithProjection>(*this);

        if (m_model) {
            cloned->m_model = m_model->clone();
        } else {
            cloned->m_request = m_request.get_compiled_model().create_infer_request();
        }

        return cloned;
    }

};

} // namespace genai
} // namespace ov
