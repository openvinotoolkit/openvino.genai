// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "gguf_tokenizer.hpp"
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
        if (candidate && genai_mode)
            genai_extension = candidate;
        if (!candidate || genai_mode)
            frontend.add_extension(object);
    };

    for (const auto& extension : extensions) {
        if (const auto* path = std::get_if<std::filesystem::path>(&extension)) {
            auto library = load_shared_object(*path);
            using CreateExtensions = void (*)(std::vector<std::shared_ptr<ov::Extension>>&);
            std::vector<std::shared_ptr<ov::Extension>> objects;
            reinterpret_cast<CreateExtensions>(get_symbol(library, "create_extensions"))(objects);
            for (const auto& object : objects) {
                // Keep the library loaded until its extension is destroyed.
                auto owner = std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<ov::Extension>>>(
                    library, object);
                add_extension(std::shared_ptr<ov::Extension>(owner, object.get()));
            }
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
