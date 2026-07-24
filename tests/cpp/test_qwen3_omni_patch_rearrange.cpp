// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// The Qwen3-Omni OV preprocessing offloads the patch reshape/transpose/flatten step to a small
// standalone graph (build_patch_rearrange_model). This test compiles that graph on CPU and asserts
// it is bit-identical to the qwen2_vl_utils host reference (pure data movement, so exact).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

#include "visual_language/clip.hpp"
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

ov::Tensor make_test_image(size_t height, size_t width, uint8_t offset) {
    ov::Tensor image(ov::element::u8, {1, height, width, 3});
    auto* data = image.data<uint8_t>();
    for (size_t i = 0; i < image.get_size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 17 + offset) % 251);
    }
    return image;
}

ov::Tensor make_host_preprocess_reference(const ov::Tensor& frame0,
                                          const ov::Tensor& frame1,
                                          size_t target_height,
                                          size_t target_width,
                                          size_t patch,
                                          size_t merge,
                                          const std::vector<float>& mean,
                                          const std::vector<float>& image_std) {
    constexpr size_t tps = 2;
    constexpr size_t channel = 3;
    ov::Tensor tiled_patches(ov::element::f32, {tps, channel, target_height, target_width});
    const ov::Tensor frames[] = {frame0, frame1};
    for (size_t i = 0; i < tps; ++i) {
        auto input = tensor_to_clip_image_u8(frames[i]);
        clip_image_u8 resized;
        bicubic_resize(input, resized, static_cast<int>(target_width), static_cast<int>(target_height));
        clip_ctx ctx;
        std::copy(mean.begin(), mean.end(), ctx.image_mean);
        std::copy(image_std.begin(), image_std.end(), ctx.image_std);
        auto patch_tensor = clip_image_f32_to_tensor(clip_image_preprocess(ctx, resized));
        std::memcpy(tiled_patches.data<float>() + i * patch_tensor.get_size(),
                    patch_tensor.data<float>(),
                    patch_tensor.get_byte_size());
    }

    const size_t grid_h = target_height / patch;
    const size_t grid_w = target_width / patch;
    auto reshaped = ov::genai::qwen2_vl_utils::reshape_image_patches(
        tiled_patches, 1, grid_h, grid_w, channel, tps, patch, merge);
    return ov::genai::qwen2_vl_utils::transpose_image_patches(reshaped);
}

ov::Tensor run_full_preprocess_on_cpu(const ov::Tensor& frame0,
                                      const ov::Tensor& frame1,
                                      size_t target_height,
                                      size_t target_width,
                                      size_t patch,
                                      size_t merge,
                                      const std::vector<float>& mean,
                                      const std::vector<float>& image_std) {
    constexpr int64_t grid_t = 1;
    constexpr int64_t tps = 2;
    constexpr int64_t channel = 3;
    const int64_t grid_h = static_cast<int64_t>(target_height / patch);
    const int64_t grid_w = static_cast<int64_t>(target_width / patch);
    const int64_t patch_i64 = static_cast<int64_t>(patch);
    const int64_t merge_i64 = static_cast<int64_t>(merge);
    std::vector<int64_t> shape8d{grid_t,
                                 tps * channel,
                                 grid_h / merge_i64,
                                 merge_i64,
                                 patch_i64,
                                 grid_w / merge_i64,
                                 merge_i64,
                                 patch_i64};
    std::vector<int64_t> shape4d{grid_t * (grid_h / merge_i64) * (grid_w / merge_i64) * merge_i64 * merge_i64,
                                 tps,
                                 channel,
                                 patch_i64 * patch_i64};
    std::vector<int64_t> shape2d{grid_t * grid_h * grid_w, channel * tps * patch_i64 * patch_i64};
    std::vector<int64_t> resize_target{static_cast<int64_t>(target_height), static_cast<int64_t>(target_width)};

    ov::Tensor image_mean(ov::element::f32, {1, 3, 1, 1});
    ov::Tensor image_scale(ov::element::f32, {1, 3, 1, 1});
    for (size_t c = 0; c < 3; ++c) {
        image_mean.data<float>()[c] = mean[c] * 255.0f;
        image_scale.data<float>()[c] = 1.0f / (image_std[c] * 255.0f);
    }

    ov::Core core;
    auto req = core.compile_model(ov::genai::qwen3_omni_testing::build_patch_preprocess_model_for_test(), "CPU")
                   .create_infer_request();
    req.set_tensor("raw_frame_0", frame0);
    req.set_tensor("raw_frame_1", frame1);
    req.set_tensor("resize_target", ov::Tensor(ov::element::i64, {2}, resize_target.data()));
    req.set_tensor("image_mean", image_mean);
    req.set_tensor("image_scale", image_scale);
    req.set_tensor("reshape_shape8d", ov::Tensor(ov::element::i64, {8}, shape8d.data()));
    req.set_tensor("reshape_shape4d", ov::Tensor(ov::element::i64, {4}, shape4d.data()));
    req.set_tensor("reshape_shape2d", ov::Tensor(ov::element::i64, {2}, shape2d.data()));
    req.infer();
    return req.get_tensor("patches_2d");
}

}  // namespace

TEST(Qwen3OmniPatchRearrange, UnsetModeUsesOvAndPropagatesCompilationFailure) {
    ScopedVisionPreprocessEnv env(nullptr);

    EXPECT_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                    "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                    {}),
                 ov::Exception);
}

TEST(Qwen3OmniPatchRearrange, EmptyModeUsesOvAndPropagatesCompilationFailure) {
    ScopedVisionPreprocessEnv env("");

    EXPECT_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                    "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                    {}),
                 ov::Exception);
}

TEST(Qwen3OmniPatchRearrange, ExplicitFullOvModePropagatesCompilationFailure) {
    ScopedVisionPreprocessEnv env("OV");

    EXPECT_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                    "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                    {}),
                 ov::Exception);
}

TEST(Qwen3OmniPatchRearrange, ExplicitOvRearrangeModePropagatesCompilationFailure) {
    ScopedVisionPreprocessEnv env("OV_REARRANGE");

    EXPECT_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                    "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                    {}),
                 ov::Exception);
}

TEST(Qwen3OmniPatchRearrange, ExplicitCppModeDoesNotCompileOnDevice) {
    ScopedVisionPreprocessEnv env("CPP");

    EXPECT_NO_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(),
                                                       "QWEN3_OMNI_TEST_INVALID_DEVICE",
                                                       {}));
}

TEST(Qwen3OmniPatchRearrange, UnknownModeIsRejected) {
    ScopedVisionPreprocessEnv env("UNKNOWN");

    EXPECT_THROW(ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(), "CPU", {}),
                 ov::Exception);
}

TEST(Qwen3OmniPatchRearrange, FullOvModelCompilesOnCpu) {
    ScopedVisionPreprocessEnv env("OV");

    EXPECT_NO_THROW(
        ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(), "CPU", {}));
}

TEST(Qwen3OmniPatchRearrange, PreprocessingFiltersUnrelatedPerModelProperties) {
    ScopedVisionPreprocessEnv env("OV");
    ov::AnyMap properties{{"MODEL_PROPERTIES",
                           ov::AnyMap{{"language_model",
                                       ov::AnyMap{{"UNSUPPORTED_LANGUAGE_ONLY_PROPERTY", true}}}}}};

    EXPECT_NO_THROW(
        ov::genai::VisionEncoderQwen3Omni(std::filesystem::temp_directory_path(), "CPU", properties));
}

TEST(Qwen3OmniPatchRearrange, FullOvInferenceMatchesHostReference) {
    constexpr size_t patch = 2;
    constexpr size_t merge = 2;
    const std::vector<float> mean{0.48145466f, 0.4578275f, 0.40821073f};
    const std::vector<float> image_std{0.26862954f, 0.26130258f, 0.27577711f};

    struct ShapeCase {
        size_t input_height;
        size_t input_width;
        size_t target_height;
        size_t target_width;
    };
    const ShapeCase cases[] = {
        {8, 12, 8, 12},   // identity resize
        {12, 20, 8, 12},  // non-square downscale
    };

    for (const auto& shape : cases) {
        auto frame0 = make_test_image(shape.input_height, shape.input_width, 3);
        auto frame1 = make_test_image(shape.input_height, shape.input_width, 29);
        auto expected = make_host_preprocess_reference(
            frame0, frame1, shape.target_height, shape.target_width, patch, merge, mean, image_std);
        auto actual = run_full_preprocess_on_cpu(
            frame0, frame1, shape.target_height, shape.target_width, patch, merge, mean, image_std);

        ASSERT_EQ(actual.get_size(), expected.get_size());
        ASSERT_EQ(actual.get_shape(),
              (ov::Shape{shape.target_height / patch * (shape.target_width / patch), 3 * 2 * patch * patch}));
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_TRUE(std::isfinite(actual.data<const float>()[i]));
            EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 0.02f)
                << "mismatch at flat index " << i << " for " << shape.input_height << "x" << shape.input_width
                << " -> " << shape.target_height << "x" << shape.target_width;
        }
    }
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
