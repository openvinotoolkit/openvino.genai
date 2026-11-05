// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/extensions.hpp"

namespace ov::genai {

inline void register_gguf_extensions(ov::frontend::gguf::FrontEnd& frontend, const ExtensionList& extensions) {
    const auto add_extension = [&](const std::shared_ptr<ov::Extension>& object) {
        OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::frontend::gguf::GenAIExtension>(object),
                        "A GenAIExtension cannot be passed through the 'extensions' property because GenAI "
                        "registers its own GGUF GenAIExtension.");
        frontend.add_extension(object);
    };

    auto& base_frontend = static_cast<ov::frontend::FrontEnd&>(frontend);
    for (const auto& extension : extensions) {
        if (const auto* path = std::get_if<std::filesystem::path>(&extension)) {
            base_frontend.add_extension(*path);
        } else {
            add_extension(std::get<std::shared_ptr<ov::Extension>>(extension));
        }
    }
}

}  // namespace ov::genai
