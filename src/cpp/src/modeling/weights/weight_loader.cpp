// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/weight_loader.hpp"

#include <openvino/core/except.hpp>

namespace {

std::string replace_once(const std::string& input, const std::string& match, const std::string& replace) {
    auto pos = input.find(match);
    if (pos == std::string::npos) {
        return input;
    }
    std::string out = input;
    out.replace(pos, match.size(), replace);
    return out;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

void default_weight_loader(WeightParameter& param,
                           WeightSource& source,
                           WeightFinalizer& finalizer,
                           const std::string& weight_name,
                           const std::optional<int>& shard_id) {
    (void)shard_id;
    if (!param.context()) {
        OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
    }
    auto weight = finalizer.finalize(weight_name, source, *param.context());
    param.bind(weight);
}

LoadReport load_model(Module& model, WeightSource& source, WeightFinalizer& finalizer, const LoadOptions& options) {
    LoadReport report;
    const auto& packed = model.packed_mapping();
    for (const auto& weight_name : source.keys()) {
        bool matched = false;
        for (const auto& rule : packed.rules) {
            if (weight_name.find(rule.match) != std::string::npos) {
                const std::string param_name = replace_once(weight_name, rule.match, rule.replace);
                WeightParameter* param = nullptr;
                try {
                    param = &model.get_parameter(param_name);
                } catch (const std::exception&) {
                    if (options.allow_unmatched) {
                        if (options.report_unmatched) {
                            report.unmatched.push_back(weight_name);
                        }
                        matched = true;
                        break;
                    }
                    throw;
                }
                if (const auto* loader = param->weight_loader()) {
                    (*loader)(*param, source, finalizer, weight_name, rule.shard_id);
                } else {
                    default_weight_loader(*param, source, finalizer, weight_name, rule.shard_id);
                }
                source.release_tensor(weight_name);
                report.matched.push_back(weight_name);
                matched = true;
                break;
            }
        }

        if (!matched) {
            WeightParameter* param = nullptr;
            try {
                param = &model.get_parameter(weight_name);
            } catch (const std::exception&) {
                if (options.allow_unmatched) {
                    if (options.report_unmatched) {
                        report.unmatched.push_back(weight_name);
                    }
                    continue;
                }
                throw;
            }
            if (const auto* loader = param->weight_loader()) {
                (*loader)(*param, source, finalizer, weight_name, std::nullopt);
            } else {
                default_weight_loader(*param, source, finalizer, weight_name, std::nullopt);
            }
            source.release_tensor(weight_name);
            report.matched.push_back(weight_name);
        }
    }
    model.finalize_parameters();

    if (options.report_missing) {
        for (auto* param : model.ctx().registered_parameters()) {
            if (!param) {
                continue;
            }
            if (!param->is_bound() && !param->is_optional()) {
                report.missing.push_back(param->name());
            }
        }
    }

    if (!options.allow_missing && !report.missing.empty()) {
        std::string message = "Missing weights for parameters:";
        for (const auto& name : report.missing) {
            message += " " + name;
        }
        OPENVINO_THROW(message);
    }
    return report;
}

void load_model(Module& model, WeightSource& source, WeightFinalizer& finalizer) {
    LoadOptions options;
    options.allow_missing = true;
    options.allow_unmatched = false;
    options.report_missing = false;
    options.report_unmatched = false;
    (void)load_model(model, source, finalizer, options);
}

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
