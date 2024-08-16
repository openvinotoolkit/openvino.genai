// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

// Inmutable LoRA Adapter that carries the adaptation matrices and the default alpha value
class OPENVINO_GENAI_EXPORTS Adapter {
    class Impl;
    std::shared_ptr<Impl> m_pimpl;
public:
    explicit Adapter(const std::string& path, float default_alpha);
    explicit Adapter(const std::string& path) {}  // alpha is calculated based on adapter file

    // TODO: Mapping between names of layers in a model and tensor names in the adapter
};


struct OPENVINO_GENAI_EXPORTS AdaptersConfig {
public:
    bool is_dynamic = false;    // false -- parameters cannot be changed during inference, this config should match for every generation
    std::vector<Adapter> adapters;
    std::vector<float> alphas;
    std::set<std::string> modules;  // additional modules that can be patched, from LoRA config "target_modules": ["q_proj", "v_proj"] etc.

    AdaptersConfig (const std::vector<Adapter>& adapters = {}) : adapters(adapters) {}
    AdaptersConfig (const Adapter& adapter) : adapters({adapters}) {}
    // AdaptersConfig& add(const Adapter& adapter);
    // AdaptersConfig& add(const Adapter& adapter, float alpha);
    AdaptersConfig& set(const Adapter& adapter, float alpha);
    AdaptersConfig& set(const Adapter& adapter);
    AdaptersConfig& remove(const Adapter);

    // Returns true if it is not a trivial config
    operator bool() const {
        return is_dynamic || !adapters.empty();
    }
};


}  // namespace genai
}  // namespace ov
