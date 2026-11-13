// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"

namespace ov::genai::bagel::npu {

// NPUW_BAGEL properties for the language, vision and VAE models, user properties applied on top
ov::AnyMap compile_properties(const ov::AnyMap& properties);

// Static sequence / KV-cache sizes NPUW_BAGEL compiles a language model variant for
class VariantNames {
public:
    VariantNames() = default;
    VariantNames(ov::Core& core, const std::string& device, const ov::AnyMap& properties);

    // Name of the static variant a language model call runs; calls without latent tokens get one dummy token
    std::string language(size_t seq_len, size_t gen_len, size_t kv_len) const;

private:
    std::vector<size_t> m_seq_buckets, m_kv_buckets;
};

}  // namespace ov::genai::bagel::npu
