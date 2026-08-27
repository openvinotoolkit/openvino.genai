// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

#include <openvino/op/transpose.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"

#include "image_generation/schedulers/ischeduler.hpp"

namespace ov::genai::video_generation_utils {

inline std::string get_class_name(const std::filesystem::path& root_dir) {
    const std::filesystem::path model_index_path = root_dir / "model_index.json";
    std::ifstream file(model_index_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
    return nlohmann::json::parse(file)["_class_name"].get<std::string>();
}

inline std::shared_ptr<IScheduler> cast_scheduler(std::shared_ptr<Scheduler>&& scheduler) {
    auto casted = std::dynamic_pointer_cast<IScheduler>(std::move(scheduler));
    OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
    return casted;
}

// Rescales the CFG noise prediction to fix overexposure when using zero terminal SNR
// noise_cfg and noise_pred_text each contain batch_size * elements_per_sample consecutive floats.
inline void rescale_noise_cfg(float* noise_cfg,
                              const float* noise_pred_text,
                              size_t batch_size,
                              size_t elements_per_sample,
                              float guidance_rescale) {
    if (elements_per_sample == 0) {
        return;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        float* cfg_sample = noise_cfg + b * elements_per_sample;
        const float* text_sample = noise_pred_text + b * elements_per_sample;

        double text_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            text_mean += text_sample[i];
        }
        text_mean /= static_cast<double>(elements_per_sample);

        double text_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = text_sample[i] - text_mean;
            text_var += diff * diff;
        }
        const float std_text = static_cast<float>(std::sqrt(text_var / static_cast<double>(elements_per_sample)));

        double cfg_mean = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            cfg_mean += cfg_sample[i];
        }
        cfg_mean /= static_cast<double>(elements_per_sample);

        double cfg_var = 0.0;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const double diff = cfg_sample[i] - cfg_mean;
            cfg_var += diff * diff;
        }
        const float std_cfg = static_cast<float>(std::sqrt(cfg_var / static_cast<double>(elements_per_sample)));

        const float scale = std_cfg > 0.0f ? std_text / std_cfg : 1.0f;
        for (size_t i = 0; i < elements_per_sample; ++i) {
            const float rescaled = cfg_sample[i] * scale;
            cfg_sample[i] = guidance_rescale * rescaled + (1.0f - guidance_rescale) * cfg_sample[i];
        }
    }
}

// Unpacked latents of shape [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p,
// p]. The patch dimensions are then permuted and collapsed into the channel dimension of shape: [B, F // p_t * H // p *
// W // p, C * p_t * p * p] (a 3 dimensional tensor). dim=0 is the batch size, dim=1 is the effective video sequence
// length, dim=2 is the effective number of input features
inline ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2),
           height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size,
                       num_channels,
                       post_patch_num_frames,
                       patch_size_t,
                       post_patch_height,
                       patch_size,
                       post_patch_width,
                       patch_size});
    std::array<int64_t, 8> order = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(ov::element::f32, {})};
    ov::op::v1::Transpose{}.evaluate(outputs,
                                     {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, order.data())});
    ov::Shape permuted_shape = outputs.at(0).get_shape();
    outputs.at(0).set_shape({permuted_shape.at(0),
                             permuted_shape.at(1) * permuted_shape.at(2) * permuted_shape.at(3),
                             permuted_shape.at(4) * permuted_shape.at(5) * permuted_shape.at(6)});
    return outputs.at(0);
}

// Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
// are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of what happens
// in the `_pack_latents` method.
inline ov::Tensor unpack_latents(const ov::Tensor& latents,
                                 size_t num_frames,
                                 size_t height,
                                 size_t width,
                                 size_t patch_size = 1,
                                 size_t patch_size_t = 1) {
    const ov::Shape in_shape = latents.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 3, "unpack_latents expects [B, S, D] input shape");
    const size_t batch_size = in_shape.at(0), sequence_length = in_shape.at(1), feature_dimensions = in_shape.at(2);

    const size_t patch_volume = patch_size_t * patch_size * patch_size;
    OPENVINO_ASSERT(feature_dimensions % patch_volume == 0, "D must be divisible by patch_size_t * patch_size * patch_size");
    const size_t num_channels = feature_dimensions / patch_volume;

    ov::Tensor reshaped{latents.get_element_type(), latents.get_shape()};
    latents.copy_to(reshaped);
    reshaped.set_shape({batch_size, num_frames, height, width, num_channels, patch_size_t, patch_size, patch_size});

    // permute(0, 4, 1, 5, 2, 6, 3, 7) -> [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    const std::array<int64_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(reshaped.get_element_type(), {})};
    ov::op::v1::Transpose{}.evaluate(
        outputs,
        {reshaped, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, const_cast<int64_t*>(order.data()))}
    );

    // (F//patch_size_t, patch_size_t) -> F, (H//patch_size, patch_size) -> H, (W//patch_size, patch_size) -> W
    const ov::Shape perm = outputs[0].get_shape(); // [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    OPENVINO_ASSERT(perm.size() == 8, "Unexpected rank after transpose");

    const size_t F = perm[2] * perm[3]; // (F//patch_size_t) * patch_size_t
    const size_t H = perm[4] * perm[5]; // (H//patch_size) * patch_size
    const size_t W = perm[6] * perm[7]; // (W//patch_size) * patch_size

    outputs[0].set_shape({perm[0], perm[1], F, H, W}); // [B, C, F, H, W]
    return outputs[0];
}

inline void reshape_to_1C111(ov::Tensor& t, size_t C) {
    size_t elems = 1;
    for (auto d : t.get_shape())
        elems *= d;

    OPENVINO_ASSERT(elems == C, "latents_mean/std must contain exactly C elements (got ", elems, ", expected ", C, ")");

    t.set_shape({1, C, 1, 1, 1});
}

inline ov::Tensor make_scalar(const ov::element::Type& et, float v) {
    ov::Tensor s(et, {});
    if (et == ov::element::f32) {
        *s.data<float>() = v;
    } else if (et == ov::element::f16) {
        *s.data<ov::float16>() = static_cast<ov::float16>(v);
    } else if (et == ov::element::bf16) {
        *s.data<ov::bfloat16>() = static_cast<ov::bfloat16>(v);
    } else {
        OPENVINO_ASSERT(false, "Unsupported element type for scalar scaling_factor");
    }
    return s;
}

// Denormalize latents across channel dim: [B, C, F, H, W]
// latents = latents * latents_std / scaling_factor + latents_mean
inline ov::Tensor denormalize_latents(const ov::Tensor& latents,
                                      ov::Tensor latents_mean,
                                      ov::Tensor latents_std,
                                      float scaling_factor = 1.0f) {
    const ov::Shape latents_shape = latents.get_shape();
    OPENVINO_ASSERT(latents_shape.size() == 5, "denormalize_latents expects [B, C, F, H, W]");
    const size_t num_channels = latents_shape[1];

    // .view(1, -1, 1, 1, 1)
    reshape_to_1C111(latents_mean, num_channels);
    reshape_to_1C111(latents_std, num_channels);

    const auto latents_type = latents.get_element_type();
    ov::Tensor scale = make_scalar(latents_type, scaling_factor);

    // latents * latents_std
    std::vector<ov::Tensor> tmp{ov::Tensor(latents_type, {})};
    ov::op::v1::Multiply{}.evaluate(tmp, {latents, latents_std});  // NUMPY broadcast

    // (...) / scaling_factor
    std::vector<ov::Tensor> tmp2{ov::Tensor(latents_type, {})};
    ov::op::v1::Divide{}.evaluate(tmp2, {tmp[0], scale});

    // (...) + latents_mean
    std::vector<ov::Tensor> result{ov::Tensor(latents_type, {})};
    ov::op::v1::Add{}.evaluate(result, {tmp2[0], latents_mean});

    return result[0];  // [B, C, F, H, W]
}

inline ov::Tensor tensor_from_vector(const std::vector<float>& data) {
    ov::Tensor t{ov::element::f32, ov::Shape{data.size()}};
    if (!data.empty()) {
        std::memcpy(t.data<float>(), data.data(), data.size() * sizeof(float));
    }
    return t;
}

}  // namespace ov::genai::video_generation_utils
