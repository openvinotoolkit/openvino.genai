// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "modeling/module.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

struct LoadOptions {
    bool allow_unmatched = false;
    bool allow_missing = true;
    bool report_unmatched = true;
    bool report_missing = true;

    static LoadOptions strict() {
        LoadOptions options;
        options.allow_unmatched = false;
        options.allow_missing = false;
        return options;
    }

    static LoadOptions lenient() {
        LoadOptions options;
        options.allow_unmatched = true;
        options.allow_missing = true;
        return options;
    }
};

struct LoadReport {
    std::vector<std::string> matched;
    std::vector<std::string> unmatched;
    std::vector<std::string> missing;
};

void default_weight_loader(WeightParameter& param,
                           WeightSource& source,
                           WeightFinalizer& finalizer,
                           const std::string& weight_name,
                           const std::optional<int>& shard_id);

LoadReport load_model(Module& model, WeightSource& source, WeightFinalizer& finalizer, const LoadOptions& options);
void load_model(Module& model, WeightSource& source, WeightFinalizer& finalizer);

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
