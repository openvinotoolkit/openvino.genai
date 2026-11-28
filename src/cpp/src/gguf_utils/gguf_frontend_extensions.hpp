// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf_tokenizer.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/extensions.hpp"

namespace ov::genai {

inline ExtensionList resolve_gguf_extensions(const ExtensionList& extensions) {
    ExtensionList resolved;
    for (const auto& extension : extensions) {
        if (const auto* path = std::get_if<std::filesystem::path>(&extension)) {
            const auto absolute_path = std::filesystem::absolute(*path);
            auto library = load_shared_object(std::filesystem::is_regular_file(absolute_path) ? absolute_path : *path);
            using CreateExtensions = void (*)(std::vector<std::shared_ptr<ov::Extension>>&);
            std::vector<std::shared_ptr<ov::Extension>> objects;
            reinterpret_cast<CreateExtensions>(get_symbol(library, "create_extensions"))(objects);
            for (const auto& object : objects) {
                auto owner =
                    std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<ov::Extension>>>(library, object);
                resolved.emplace_back(std::shared_ptr<ov::Extension>(owner, object.get()));
            }
        } else {
            resolved.push_back(extension);
        }
    }
    return resolved;
}

inline bool is_gguf_frontend_extension(const std::shared_ptr<ov::Extension>& object) {
    return std::dynamic_pointer_cast<ov::frontend::ConversionExtensionBase>(object) ||
           std::dynamic_pointer_cast<ov::frontend::DecoderTransformationExtension>(object) ||
           std::dynamic_pointer_cast<ov::frontend::TelemetryExtension>(object) ||
           std::dynamic_pointer_cast<ov::frontend::gguf::ArchitectureExtension>(object);
}

inline void register_gguf_extensions(ov::frontend::FrontEnd& frontend, const ExtensionList& extensions) {
    for (const auto& extension : resolve_gguf_extensions(extensions)) {
        frontend.add_extension(std::get<std::shared_ptr<ov::Extension>>(extension));
    }
}

}  // namespace ov::genai
