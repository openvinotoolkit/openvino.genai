// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// The Qwen3-Omni GPU preprocessing offloads the patch reshape/transpose/flatten step to a small
// standalone graph (build_patch_rearrange_model). This test compiles that graph on CPU and asserts
// it is bit-identical to the qwen2_vl_utils host reference (pure data movement, so exact).

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen3_omni/classes.hpp"

namespace {

class ScopedVisionPreprocessEnv {
public:
    explicit ScopedVisionPreprocessEnv(const char* value) {
        if (const char* previous = std::getenv("VISION_PREPROCESS")) {
            m_previous = previous;
        }
        set(value);
    }

    ~ScopedVisionPreprocessEnv() {
        set(m_previous ? m_previous->c_str() : nullptr);
    }

private:
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("VISION_PREPROCESS", value ? value : "");
#else
        if (value) {
            setenv("VISION_PREPROCESS", value, 1);
        } else {
            unsetenv("VISION_PREPROCESS");
        }
#endif
    }

    std::optional<std::string> m_previous;
};

}  // namespace

TEST(Qwen3OmniPatchRearrange, FallsBackToHostWhenDeviceCompilationFails) {
    ScopedVisionPreprocessEnv env("GPU");

    EXPECT_NO_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                       "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                       {}));
}

TEST(Qwen3OmniPatchRearrange, MatchesQwen2VlReference) {
    // Non-square grid so a transpose-permutation bug can't hide behind symmetry.
    const size_t grid_t = 1, tps = 2, channel = 3, patch = 4, merge = 2, grid_h = 4, grid_w = 6;
    const size_t H = grid_h * patch, W = grid_w * patch;

    // tiled_patches: f32 {tps, channel, H, W} filled with a deterministic ramp.
    ov::Tensor tiled_patches(ov::element::f32, ov::Shape{tps, channel, H, W});
    float* p = tiled_patches.data<float>();
    for (size_t i = 0; i < tiled_patches.get_size(); ++i) {
        p[i] = static_cast<float>(i);
    }

    // Reference: reshape -> transpose (the final flatten is a pure reshape, so we compare flat data).
    ov::Tensor reshaped = ov::genai::qwen2_vl_utils::reshape_image_patches(
        tiled_patches, grid_t, grid_h, grid_w, channel, tps, patch, merge);
    ov::Tensor ref = ov::genai::qwen2_vl_utils::transpose_image_patches(reshaped);

    // Same shape formula as preprocess_to_patches.
    std::vector<int64_t> shape8d{(int64_t)grid_t, (int64_t)(tps * channel), (int64_t)(grid_h / merge), (int64_t)merge,
                                 (int64_t)patch,  (int64_t)(grid_w / merge), (int64_t)merge,           (int64_t)patch};
    std::vector<int64_t> shape4d{(int64_t)(grid_t * (grid_h / merge) * (grid_w / merge) * (merge * merge)),
                                 (int64_t)tps, (int64_t)channel, (int64_t)(patch * patch)};
    std::vector<int64_t> shape2d{(int64_t)(grid_t * grid_h * grid_w), (int64_t)(channel * tps * patch * patch)};

    // Run the production graph on CPU.
    ov::Core core;
    auto req = core.compile_model(ov::genai::qwen3_omni_testing::build_patch_rearrange_model_for_test(), "CPU")
                   .create_infer_request();
    req.set_tensor("tiled_patches", tiled_patches);
    req.set_tensor("reshape_shape8d", ov::Tensor(ov::element::i64, ov::Shape{8}, shape8d.data()));
    req.set_tensor("reshape_shape4d", ov::Tensor(ov::element::i64, ov::Shape{4}, shape4d.data()));
    req.set_tensor("reshape_shape2d", ov::Tensor(ov::element::i64, ov::Shape{2}, shape2d.data()));
    req.infer();
    const ov::Tensor out = req.get_tensor("patches_2d");

    ASSERT_EQ(out.get_size(), ref.get_size());
    const float* out_data = out.data<float>();
    const float* ref_data = ref.data<float>();
    for (size_t i = 0; i < ref.get_size(); ++i) {
        EXPECT_EQ(out_data[i], ref_data[i]) << "mismatch at flat index " << i;
    }
}
