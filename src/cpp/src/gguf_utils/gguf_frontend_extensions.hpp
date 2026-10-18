// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/extensions.hpp"

namespace ov::genai {

inline std::shared_ptr<ov::frontend::gguf::GenAIExtension> register_gguf_extensions(
    ov::frontend::gguf::FrontEnd& frontend,
    const ExtensionList& extensions,
    std::optional<ov::frontend::gguf::GenAIExtension::InputMode> genai_mode = std::nullopt) {
    std::shared_ptr<ov::frontend::gguf::GenAIExtension> genai_extension;
    const auto add_extension = [&](const std::shared_ptr<ov::Extension>& object) {
        auto candidate = std::dynamic_pointer_cast<ov::frontend::gguf::GenAIExtension>(object);
        if (const auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(object))
            candidate = std::dynamic_pointer_cast<ov::frontend::gguf::GenAIExtension>(so_extension->extension());
        if (candidate && genai_mode)
            genai_extension = candidate;
        if (!candidate || genai_mode)
            frontend.add_extension(object);
    };

    for (const auto& extension : extensions) {
        if (const auto* path = std::get_if<std::filesystem::path>(&extension)) {
            for (const auto& object : ov::detail::load_extensions(*path))
                add_extension(object);
        } else {
            add_extension(std::get<std::shared_ptr<ov::Extension>>(extension));
        }
    }

    if (genai_mode && !genai_extension) {
        genai_extension = std::make_shared<ov::frontend::gguf::GenAIExtension>(*genai_mode);
        frontend.add_extension(genai_extension);
    }
    return genai_extension;
}

}  // namespace ov::genai
