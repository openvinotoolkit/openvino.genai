// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bagel/npu/npu_config.hpp"

#include <algorithm>
#include <sstream>

namespace ov::genai::bagel::npu {
namespace {

std::vector<size_t> parse_buckets(const std::string& text) {
    std::vector<size_t> buckets;
    std::stringstream ss(text);
    for (std::string item; std::getline(ss, item, ',');) {
        buckets.push_back(std::stoul(item));
    }
    return buckets;
}

size_t pick_bucket(const std::vector<size_t>& buckets, size_t n) {
    const auto it = std::find_if(buckets.begin(), buckets.end(), [n](size_t b) {
        return b >= n;
    });
    return it == buckets.end() ? n : *it;
}

}  // namespace

ov::AnyMap compile_properties(const ov::AnyMap& properties) {
    ov::AnyMap config{{"NPU_USE_NPUW", "YES"}, {"NPUW_BAGEL", "YES"}};
    for (const auto& [key, value] : properties) {
        config[key] = value;
    }
    return config;
}

VariantNames::VariantNames(ov::Core& core, const std::string& device, const ov::AnyMap& properties) {
    const auto buckets = [&](const std::string& key) {
        const auto it = properties.find(key);
        return parse_buckets(it != properties.end() ? it->second.as<std::string>()
                                                    : core.get_property(device, key).as<std::string>());
    };
    m_seq_buckets = buckets("NPUW_BAGEL_SEQ_BUCKETS");
    m_kv_buckets = buckets("NPUW_BAGEL_KV_BUCKETS");
}

std::string VariantNames::language(size_t seq_len, size_t gen_len, size_t kv_len) const {
    if (gen_len == 0) {
        seq_len = pick_bucket(m_seq_buckets, seq_len + 1);
        gen_len = 1;
    }
    return "language S=" + std::to_string(seq_len) + " gen=" + std::to_string(gen_len) +
           " kv=" + std::to_string(pick_bucket(m_kv_buckets, std::max<size_t>(kv_len, 1)));
}

}  // namespace ov::genai::bagel::npu
