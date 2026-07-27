// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_omni/classes.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "utils.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/vl_sdpa_transformations.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::genai {

namespace {

VisionEncoderQwen3Omni::PatchPreprocMode parse_preproc_mode_env() {
    const char* env = std::getenv("VISION_PREPROCESS");
    if (!env || env[0] == '\0') {
        return VisionEncoderQwen3Omni::PatchPreprocMode::OV;
    }

    const std::string value(env);
    if (value == "CPP") {
        return VisionEncoderQwen3Omni::PatchPreprocMode::CPP;
    }
    if (value == "OV_REARRANGE") {
        return VisionEncoderQwen3Omni::PatchPreprocMode::OV_REARRANGE;
    }
    if (value == "OV") {
        return VisionEncoderQwen3Omni::PatchPreprocMode::OV;
    }

    OPENVINO_THROW("Unsupported VISION_PREPROCESS value: ",
                   value,
                   ". Expected OV, OV_REARRANGE, CPP, or an empty value for the default OV implementation.");
}

// Append the patch reshape/transpose/flatten tail (bit-identical to
// qwen2_vl_utils::reshape_image_patches + transpose_image_patches + flatten). The 8D/4D
// transpose decomposition mirrors the validated qwen2vl in-graph preprocessing.
//   Input  : f32 NCHW {temporal_patch_size, channel, H, W}
//   Output : f32 {grid_t*grid_h*grid_w, channel*temporal_patch_size*patch*patch}
std::shared_ptr<ov::Node> append_patch_rearrange(const std::shared_ptr<ov::Node>& patches,
                                                 const std::shared_ptr<ov::Node>& reshape_shape8d,
                                                 const std::shared_ptr<ov::Node>& reshape_shape4d,
                                                 const std::shared_ptr<ov::Node>& reshape_shape2d) {
    auto reshaped8d = std::make_shared<ov::op::v1::Reshape>(patches, reshape_shape8d, true);
    auto perm8 =
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{8}, std::vector<int32_t>{0, 2, 5, 3, 6, 1, 4, 7});
    auto transposed8d = std::make_shared<ov::op::v1::Transpose>(reshaped8d, perm8);

    auto reshaped4d = std::make_shared<ov::op::v1::Reshape>(transposed8d, reshape_shape4d, true);
    auto perm4 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto transposed4d = std::make_shared<ov::op::v1::Transpose>(reshaped4d, perm4);

    return std::make_shared<ov::op::v1::Reshape>(transposed4d, reshape_shape2d, true);
}

// Bicubic resize matching the CPU clip bicubic_resize semantics. The CPU path uses Pillow's
// antialiased bicubic (kernel a=-0.5, support widened by the downscale factor); the matching
// OpenVINO op mode is BICUBIC_PILLOW, which the GPU plugin implements via resample_kernel_pil_ref
// with the identical filter/support logic. Plain CUBIC (4-tap, no antialias) diverges on
// downscale and loses fine detail, so it must not be used here. For PILLOW modes the kernel
// ignores coordinate_transformation_mode/cube_coeff and is intrinsically antialiased.
std::shared_ptr<ov::Node> create_bicubic_resize(const std::shared_ptr<ov::Node>& input,
                                                const std::shared_ptr<ov::Node>& target_size) {
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.cube_coeff = -0.5f;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = true;
    return std::make_shared<ov::op::v11::Interpolate>(input, target_size, axes, attrs);
}

// Resize + normalize a single raw u8 NHWC frame into a normalized f32 NCHW {1,channel,H,W} tensor.
//   normalize: clamp(0,255) -> (x - mean*255) * 1/(std*255)  ==  (x/255 - mean)/std
std::shared_ptr<ov::Node> create_resize_normalize(const std::shared_ptr<ov::Node>& raw_frame,
                                                  const std::shared_ptr<ov::Node>& resize_target,
                                                  const std::shared_ptr<ov::Node>& image_mean,
                                                  const std::shared_ptr<ov::Node>& image_scale) {
    auto f32 = std::make_shared<ov::op::v0::Convert>(raw_frame, ov::element::f32);
    auto nchw = std::make_shared<ov::op::v1::Transpose>(
        f32,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    auto resized = create_bicubic_resize(nchw, resize_target);
    auto clamped = std::make_shared<ov::op::v0::Clamp>(resized, 0.0, 255.0);
    auto centered = std::make_shared<ov::op::v1::Subtract>(clamped, image_mean);
    return std::make_shared<ov::op::v1::Multiply>(centered, image_scale);
}

// Build a small standalone model that performs the patch reshape/transpose/flatten
// (bit-identical to qwen2_vl_utils::reshape_image_patches + transpose_image_patches),
// so this pure data-movement step runs on the accelerator instead of the host CPU.
//   Input  : "tiled_patches" f32 NCHW {temporal_patch_size, channel, H, W} (already resized+normalized)
//   Inputs : "reshape_shape8d"/"reshape_shape4d"/"reshape_shape2d" i64 target shapes (per image)
//   Output : "patches_2d" f32 {grid_t*grid_h*grid_w, channel*temporal_patch_size*patch*patch}
std::shared_ptr<ov::Model> build_patch_rearrange_model() {
    auto patches = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    patches->set_friendly_name("tiled_patches");
    patches->output(0).get_tensor().set_names({"tiled_patches"});

    auto reshape_shape8d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{8});
    reshape_shape8d->set_friendly_name("reshape_shape8d");
    reshape_shape8d->output(0).get_tensor().set_names({"reshape_shape8d"});

    auto reshape_shape4d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    reshape_shape4d->set_friendly_name("reshape_shape4d");
    reshape_shape4d->output(0).get_tensor().set_names({"reshape_shape4d"});

    auto reshape_shape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    reshape_shape2d->set_friendly_name("reshape_shape2d");
    reshape_shape2d->output(0).get_tensor().set_names({"reshape_shape2d"});

    auto flattened = append_patch_rearrange(patches, reshape_shape8d, reshape_shape4d, reshape_shape2d);
    auto result = std::make_shared<ov::op::v0::Result>(flattened);
    result->output(0).get_tensor().set_names({"patches_2d"});

    return std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{patches, reshape_shape8d, reshape_shape4d, reshape_shape2d},
        "qwen3_omni_patch_rearrange");
}

// Build the full preprocessing model (Stage 2): raw u8 frame(s) -> resize + normalize ->
// reshape/transpose/flatten. The temporal pair is always assembled from two raw frames
// (for images the same frame is fed twice, matching the CPU loop that resizes the image
// twice into the two temporal slots).
//   Inputs : "raw_frame_0"/"raw_frame_1" u8 NHWC {1,H0,W0,3}, "resize_target" i64{2} {H,W},
//            "image_mean"/"image_scale" f32 {1,3,1,1} normalization constants (mean*255, 1/(std*255)),
//            "reshape_shape8d"/"reshape_shape4d"/"reshape_shape2d" i64 target shapes
//   Output : "patches_2d" f32 {grid_t*grid_h*grid_w, channel*temporal_patch_size*patch*patch}
// image_mean/image_scale are runtime Parameters (not baked Constants) so the same compiled model
// serves both the image and video ProcessorConfig: the caller feeds the per-call normalization,
// matching the host CPU/Stage-1 paths which read the config at runtime.
std::shared_ptr<ov::Model> build_patch_preprocess_model() {
    auto image_mean = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 1, 1});
    image_mean->set_friendly_name("image_mean");
    image_mean->output(0).get_tensor().set_names({"image_mean"});

    auto image_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 1, 1});
    image_scale->set_friendly_name("image_scale");
    image_scale->output(0).get_tensor().set_names({"image_scale"});

    auto raw_frame_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, -1, -1, 3});
    raw_frame_0->set_friendly_name("raw_frame_0");
    raw_frame_0->output(0).get_tensor().set_names({"raw_frame_0"});

    auto raw_frame_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, -1, -1, 3});
    raw_frame_1->set_friendly_name("raw_frame_1");
    raw_frame_1->output(0).get_tensor().set_names({"raw_frame_1"});

    auto resize_target = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    resize_target->set_friendly_name("resize_target");
    resize_target->output(0).get_tensor().set_names({"resize_target"});

    auto reshape_shape8d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{8});
    reshape_shape8d->set_friendly_name("reshape_shape8d");
    reshape_shape8d->output(0).get_tensor().set_names({"reshape_shape8d"});

    auto reshape_shape4d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    reshape_shape4d->set_friendly_name("reshape_shape4d");
    reshape_shape4d->output(0).get_tensor().set_names({"reshape_shape4d"});

    auto reshape_shape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    reshape_shape2d->set_friendly_name("reshape_shape2d");
    reshape_shape2d->output(0).get_tensor().set_names({"reshape_shape2d"});

    auto frame0 = create_resize_normalize(raw_frame_0, resize_target, image_mean, image_scale);
    auto frame1 = create_resize_normalize(raw_frame_1, resize_target, image_mean, image_scale);
    auto temporal =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{frame0->output(0), frame1->output(0)}, 0);

    auto flattened = append_patch_rearrange(temporal, reshape_shape8d, reshape_shape4d, reshape_shape2d);
    auto result = std::make_shared<ov::op::v0::Result>(flattened);
    result->output(0).get_tensor().set_names({"patches_2d"});

    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{raw_frame_0,
                                                           raw_frame_1,
                                                           resize_target,
                                                           image_mean,
                                                           image_scale,
                                                           reshape_shape8d,
                                                           reshape_shape4d,
                                                           reshape_shape2d},
                                       "qwen3_omni_patch_preprocess");
}

}  // namespace

namespace qwen3_omni_testing {
// Test-only accessor for the anonymous-namespace patch rearrange builder.
std::shared_ptr<ov::Model> build_patch_rearrange_model_for_test() {
    return build_patch_rearrange_model();
}

std::shared_ptr<ov::Model> build_patch_preprocess_model_for_test() {
    return build_patch_preprocess_model();
}
}  // namespace qwen3_omni_testing

// --- VisionEncoderQwen3Omni ---

VisionEncoderQwen3Omni::VisionEncoderQwen3Omni(const std::filesystem::path& model_dir,
                                               const std::string& device,
                                               const ov::AnyMap properties)
    : VisionEncoderQwen3VL(model_dir, ConfigOnlyTag{}) {
    initialize_patch_preprocessing(device, properties);
}

VisionEncoderQwen3Omni::VisionEncoderQwen3Omni(const ModelsMap& models_map,
                                               const std::filesystem::path& config_dir_path,
                                               const std::string& device,
                                               const ov::AnyMap properties)
    : VisionEncoderQwen3VL(models_map, config_dir_path, ConfigOnlyTag{}) {
    initialize_patch_preprocessing(device, properties);
}

void VisionEncoderQwen3Omni::initialize_patch_preprocessing(const std::string& device,
                                                             const ov::AnyMap& properties) {
    // Vision encoder runs no transformer inference, but a tiny model offloads patch
    // preprocessing (resize/normalize and/or reshape/transpose/flatten) to `device`.
    m_preproc_mode = parse_preproc_mode_env();

    if (m_preproc_mode == PatchPreprocMode::CPP) {
        return;
    }

    // Preserve the requested device and surface compilation errors instead of silently changing paths.
    auto model = m_preproc_mode == PatchPreprocMode::OV ? build_patch_preprocess_model()
                                                        : build_patch_rearrange_model();
    auto compiled = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    m_ireq_queue_patch_rearrange = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest { return compiled.create_infer_request(); });
}

void VisionEncoderQwen3Omni::preprocess_to_patches(const std::vector<ov::Tensor>& images,
                                                   const ProcessorConfig& config,
                                                   ov::Tensor& out_tensor,
                                                   ImageSize& out_rsz_size,
                                                   size_t frame_num,
                                                   size_t frame_id) {
    OPENVINO_ASSERT(config.temporal_patch_size == 2u, "temporal_patch_size != 2.");
    if (images.size() > 1) {
        OPENVINO_ASSERT(config.temporal_patch_size == images.size(), "temporal_patch_size != images.size()");
    }

    const auto& orig_shape = images[0].get_shape();
    const auto target_image_size = qwen2_vl_utils::smart_resize(orig_shape.at(1),
                                                                orig_shape.at(2),
                                                                config.patch_size * config.merge_size,
                                                                config.min_pixels,
                                                                config.max_pixels);

    const size_t channel = 3;
    const auto merge = config.merge_size;
    const auto patch = config.patch_size;
    const auto tps = config.temporal_patch_size;
    const size_t grid_t = 1;
    const auto grid_h = target_image_size.height / patch;
    const auto grid_w = target_image_size.width / patch;

    // Per-image target shapes for the in-graph reshape/transpose (mirror the validated qwen2vl
    // in-graph decomposition: 8D transpose, 4D transpose, 2D flatten).
    int64_t shape8d[8] = {static_cast<int64_t>(grid_t),
                          static_cast<int64_t>(tps * channel),
                          static_cast<int64_t>(grid_h / merge),
                          static_cast<int64_t>(merge),
                          static_cast<int64_t>(patch),
                          static_cast<int64_t>(grid_w / merge),
                          static_cast<int64_t>(merge),
                          static_cast<int64_t>(patch)};
    int64_t shape4d[4] = {static_cast<int64_t>(grid_t * (grid_h / merge) * (grid_w / merge) * (merge * merge)),
                          static_cast<int64_t>(tps),
                          static_cast<int64_t>(channel),
                          static_cast<int64_t>(patch * patch)};
    int64_t shape2d[2] = {static_cast<int64_t>(grid_t * grid_h * grid_w),
                          static_cast<int64_t>(channel * tps * patch * patch)};

    ov::Tensor shape8d_tensor(ov::element::i64, ov::Shape{8});
    ov::Tensor shape4d_tensor(ov::element::i64, ov::Shape{4});
    ov::Tensor shape2d_tensor(ov::element::i64, ov::Shape{2});
    std::memcpy(shape8d_tensor.data(), shape8d, sizeof(shape8d));
    std::memcpy(shape4d_tensor.data(), shape4d, sizeof(shape4d));
    std::memcpy(shape2d_tensor.data(), shape2d, sizeof(shape2d));

    auto copy_to_output = [&](const ov::Tensor& patches) {
        const auto& patches_shape = patches.get_shape();
        if (frame_id == 0u) {
            auto out_shape = patches_shape;
            out_shape[0] = patches_shape[0] * frame_num;
            out_tensor = ov::Tensor(patches.get_element_type(), out_shape);
        }
        std::memcpy(reinterpret_cast<uint8_t*>(out_tensor.data()) + frame_id * patches.get_byte_size(),
                    patches.data(),
                    patches.get_byte_size());
    };

    ov::Tensor flattened;
    if (m_preproc_mode == PatchPreprocMode::OV) {
        // Stage 2: upload raw u8 frame(s) and run resize + normalize + reshape/transpose/flatten on the OV device.
        // For an image the same frame fills both temporal slots (matching the CPU loop, which resizes
        // the image twice). For video the two adjacent frames fill the slots.
        ov::Tensor resize_target_tensor(ov::element::i64, ov::Shape{2});
        int64_t* resize_target = resize_target_tensor.data<int64_t>();
        resize_target[0] = static_cast<int64_t>(target_image_size.height);
        resize_target[1] = static_cast<int64_t>(target_image_size.width);

        // Normalization constants from the per-call config (image vs video), matching the host
        // CPU/Stage-1 paths: centered = clamp(x) - mean*255, scaled = centered * 1/(std*255).
        ov::Tensor image_mean_tensor(ov::element::f32, ov::Shape{1, 3, 1, 1});
        ov::Tensor image_scale_tensor(ov::element::f32, ov::Shape{1, 3, 1, 1});
        float* image_mean = image_mean_tensor.data<float>();
        float* image_scale = image_scale_tensor.data<float>();
        for (size_t c = 0; c < 3; ++c) {
            image_mean[c] = config.image_mean[c] * 255.0f;
            image_scale[c] = 1.0f / (config.image_std[c] * 255.0f);
        }

        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_patch_rearrange.get());
        auto& ireq = guard.get();
        ireq.set_tensor("raw_frame_0", images[0]);
        ireq.set_tensor("raw_frame_1", images.size() > 1 ? images[1] : images[0]);
        ireq.set_tensor("resize_target", resize_target_tensor);
        ireq.set_tensor("image_mean", image_mean_tensor);
        ireq.set_tensor("image_scale", image_scale_tensor);
        ireq.set_tensor("reshape_shape8d", shape8d_tensor);
        ireq.set_tensor("reshape_shape4d", shape4d_tensor);
        ireq.set_tensor("reshape_shape2d", shape2d_tensor);
        ireq.infer();
        const auto& out = ireq.get_tensor("patches_2d");
        copy_to_output(out);
        out_rsz_size = ImageSize{grid_h, grid_w};
        return;
    } else {
        // CPU resize + normalize into the temporal-stacked f32 NCHW patches.
        ov::Tensor tiled_patches(ov::element::f32,
                                 {tps, channel, target_image_size.height, target_image_size.width});
        for (size_t i = 0; i < tps; i++) {
            const auto& image = images.size() > i ? images[i] : images[0];
            auto input_image = tensor_to_clip_image_u8(image);
            clip_image_u8 resized_image;
            bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);
            clip_ctx ctx;
            std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
            std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
            auto normalized_image = clip_image_preprocess(ctx, resized_image);
            auto patch_tensor = clip_image_f32_to_tensor(normalized_image);
            std::memcpy(tiled_patches.data<float>() + i * patch_tensor.get_byte_size() / sizeof(float),
                        patch_tensor.data<float>(),
                        patch_tensor.get_byte_size());
        }

        if (m_preproc_mode == PatchPreprocMode::OV_REARRANGE) {
            // Stage 1: OV-device reshape/transpose/flatten only (bit-identical to the host path).
            CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_patch_rearrange.get());
            auto& ireq = guard.get();
            ireq.set_tensor("tiled_patches", tiled_patches);
            ireq.set_tensor("reshape_shape8d", shape8d_tensor);
            ireq.set_tensor("reshape_shape4d", shape4d_tensor);
            ireq.set_tensor("reshape_shape2d", shape2d_tensor);
            ireq.infer();
            const auto& out = ireq.get_tensor("patches_2d");
            copy_to_output(out);
            out_rsz_size = ImageSize{grid_h, grid_w};
            return;
        } else {
            // Reference host reshape/transpose/flatten (VISION_PREPROCESS=CPP).
            auto reshaped = qwen2_vl_utils::reshape_image_patches(tiled_patches,
                                                                  grid_t,
                                                                  grid_h,
                                                                  grid_w,
                                                                  channel,
                                                                  tps,
                                                                  patch,
                                                                  merge);
            auto transposed = qwen2_vl_utils::transpose_image_patches(reshaped);
            ov::Shape cpu_flat_shape{grid_t * grid_h * grid_w, channel * tps * patch * patch};
            flattened = ov::Tensor(transposed.get_element_type(), cpu_flat_shape);
            std::memcpy(flattened.data(), transposed.data(), transposed.get_byte_size());
        }
    }
    copy_to_output(flattened);
    out_rsz_size = ImageSize{grid_h, grid_w};
}

EncodedImage VisionEncoderQwen3Omni::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    (void)config_map;  // Required by interface but not used in this implementation
    EncodedImage encoded_img;
    preprocess_to_patches({image},
                          m_processor_config,
                          encoded_img.resized_source,
                          encoded_img.resized_source_size,
                          1,
                          0);
    return encoded_img;
}

EncodedVideo VisionEncoderQwen3Omni::encode_frames(const std::vector<ov::Tensor>& frames) {
    EncodedVideo encoded_video;
    const auto& config = m_video_processor_config;

    const auto frames_size = frames.size();
    encoded_video.frame_num = (frames_size + config.temporal_patch_size - 1) / config.temporal_patch_size;

    size_t frame_id = 0;
    size_t i = 0;
    for (; i + config.temporal_patch_size <= frames_size; i += config.temporal_patch_size) {
        preprocess_to_patches(std::vector<ov::Tensor>(frames.begin() + i,
                                                      frames.begin() + i + config.temporal_patch_size),
                              config,
                              encoded_video.video_features,
                              encoded_video.resized_source_size,
                              encoded_video.frame_num,
                              frame_id);
        frame_id++;
    }
    for (; i < frames_size; i++) {
        preprocess_to_patches({frames[i]},
                              config,
                              encoded_video.video_features,
                              encoded_video.resized_source_size,
                              encoded_video.frame_num,
                              frame_id);
        frame_id++;
    }
    return encoded_video;
}

// --- InputsEmbedderQwen3Omni ---

InputsEmbedderQwen3Omni::InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                                                 const std::filesystem::path& model_dir,
                                                 const std::string& device,
                                                 const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, model_dir, device, device_config),
      m_audio_token_id(vlm_config.audio_token_id) {
    // Audio encoder is optional — check is_available() / has_audio_encoder() before encoding
    m_audio_encoder = std::make_unique<AudioEncoderQwen3Omni>(model_dir, vlm_config, device, device_config);

    // Merged vision model is optional — check has_merged_vision_model() before calling
    // run_video_image_embeddings_merger(). Without it the model works as text+audio only.
    auto vision_model_path = model_dir / "openvino_vision_embeddings_model.xml";
    if (std::filesystem::exists(vision_model_path)) {
        auto model = utils::singleton_core().read_model(vision_model_path);
        // Enable VLSDPA (CM flash-attention) fusion on the vision transformer's self-attention.
        // Sets rt_info "model_type_hint"="QWenVL" so the GPU SDPAToVLSDPA pass can replace the dense
        // "attention_mask" parameter with a packed "cu_seq_lens" input (eliminates N*N score I/O).
        utils::request_vl_sdpa_transformations(model);
        auto compiled = utils::singleton_core().compile_model(
            model, device, utils::get_model_properties(device_config, "vision_embeddings", device));
        m_with_cu_seqlens_input = utils::has_vl_sdpa_input(compiled, "cu_seq_lens");
        ov::genai::utils::print_compiled_model_properties(compiled,
            m_with_cu_seqlens_input ? "Omni vision embeddings model with VLSDPA optimization ENABLED"
                                    : "Omni vision embeddings model with VLSDPA optimization DISABLED");
        m_ireq_queue_merged_vision = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
        // Cache rotary dim to avoid queue lock in get_rotary_pos_emb
        auto rotary_pshape = compiled.input("rotary_pos_emb").get_partial_shape();
        m_rotary_dim = static_cast<size_t>(rotary_pshape[rotary_pshape.rank().get_length() - 1].get_length());
    }
}

InputsEmbedderQwen3Omni::InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                                                 const ModelsMap& models_map,
                                                 const Tokenizer& tokenizer,
                                                 const std::filesystem::path& config_dir_path,
                                                 const std::string& device,
                                                 const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config),
      m_audio_token_id(vlm_config.audio_token_id) {
    // Audio encoder is optional — check is_available() / has_audio_encoder() before encoding
    m_audio_encoder = std::make_unique<AudioEncoderQwen3Omni>(config_dir_path, vlm_config, device, device_config);

    // Merged vision model is optional — check has_merged_vision_model() before calling
    // run_video_image_embeddings_merger(). Without it the model works as text+audio only.
    if (models_map.count("vision_embeddings")) {
        const auto& [model_str, weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto model = utils::singleton_core().read_model(model_str, weights);
        // Enable VLSDPA (CM flash-attention) fusion on the vision transformer's self-attention.
        // Sets rt_info "model_type_hint"="QWenVL" so the GPU SDPAToVLSDPA pass can replace the dense
        // "attention_mask" parameter with a packed "cu_seq_lens" input (eliminates N*N score I/O).
        utils::request_vl_sdpa_transformations(model);
        auto compiled = utils::singleton_core().compile_model(
            model, device, utils::get_model_properties(device_config, "vision_embeddings", device));
        m_with_cu_seqlens_input = utils::has_vl_sdpa_input(compiled, "cu_seq_lens");
        ov::genai::utils::print_compiled_model_properties(compiled,
            m_with_cu_seqlens_input ? "Omni vision embeddings model with VLSDPA optimization ENABLED"
                                    : "Omni vision embeddings model with VLSDPA optimization DISABLED");
        m_ireq_queue_merged_vision = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
        // Cache rotary dim to avoid queue lock in get_rotary_pos_emb
        auto rotary_pshape = compiled.input("rotary_pos_emb").get_partial_shape();
        m_rotary_dim = static_cast<size_t>(rotary_pshape[rotary_pshape.rank().get_length() - 1].get_length());
    }
}

void InputsEmbedderQwen3Omni::encode_audios(const std::vector<ov::Tensor>& audios) {
    if (audios.empty() || !has_audio_encoder()) {
        m_audio_embeddings = ov::Tensor();
        return;
    }

    std::vector<ov::Tensor> all_features;
    size_t total_tokens = 0;
    size_t hidden_size = 0;

    for (const auto& audio : audios) {
        // Empty audio tensors are silently skipped so callers can pass placeholders.
        if (audio.get_size() == 0) {
            continue;
        }

        auto features = m_audio_encoder->encode(audio);

        const ov::Shape& feature_shape = features.get_shape();
        OPENVINO_ASSERT(feature_shape.size() == 2,
                        "Audio encoder output must be rank-2 [num_tokens, hidden_size], got shape rank ",
                        feature_shape.size());
        OPENVINO_ASSERT(features.get_element_type() == ov::element::f32,
                        "Audio encoder output element type must be f32, got ",
                        features.get_element_type());

        const size_t num_tokens = feature_shape[0];
        const size_t current_hidden_size = feature_shape[1];
        if (all_features.empty()) {
            hidden_size = current_hidden_size;
        } else {
            OPENVINO_ASSERT(current_hidden_size == hidden_size,
                            "All audio feature tensors must have the same hidden_size. Expected ",
                            hidden_size,
                            ", got ",
                            current_hidden_size);
        }

        total_tokens += num_tokens;
        all_features.push_back(features);
    }

    m_audio_embeddings = ov::Tensor(ov::element::f32, {total_tokens, hidden_size});
    auto* dst = m_audio_embeddings.data<float>();
    for (const auto& feat : all_features) {
        auto byte_size = feat.get_byte_size();
        std::memcpy(dst, feat.data<float>(), byte_size);
        dst += feat.get_size();
    }
}

ov::Tensor InputsEmbedderQwen3Omni::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    auto input_embeds = InputsEmbedderQwen3VL::get_inputs_embeds(prompt,
                                                                 images,
                                                                 videos,
                                                                 metrics,
                                                                 recalculate_merged_embeddings,
                                                                 image_sequence,
                                                                 videos_sequence,
                                                                 history_vision_count);

    // Capture input_ids set by parent's get_inputs_embeds() into a local variable,
    // making the cross-class data dependency explicit rather than relying on
    // implicit ordering of m_last_input_ids population.
    std::vector<int64_t> input_ids_vec(m_last_input_ids.data<int64_t>(),
                                       m_last_input_ids.data<int64_t>() + m_last_input_ids.get_size());

    // If we have audio embeddings, replace audio token positions
    if (m_audio_embeddings && m_audio_embeddings.get_size() > 0 && m_audio_token_id >= 0) {
        merge_audio_embeddings(input_embeds, input_ids_vec);
    }

    return input_embeds;
}

void InputsEmbedderQwen3Omni::merge_audio_embeddings(ov::Tensor& input_embeds, const std::vector<int64_t>& input_ids) {
    if (!m_audio_embeddings || m_audio_embeddings.get_size() == 0) {
        return;
    }

    const auto& shape = input_embeds.get_shape();
    const auto seq_len = shape[1];
    const auto hidden_size = shape[2];

    const auto audio_hidden_size = m_audio_embeddings.get_shape()[1];
    OPENVINO_ASSERT(audio_hidden_size == hidden_size,
                    "Audio embedding hidden_size (",
                    audio_hidden_size,
                    ") must match input embedding hidden_size (",
                    hidden_size,
                    "). Check that audio encoder output dimension matches the language model.");

    OPENVINO_ASSERT(input_ids.size() >= seq_len,
                    "input_ids size (",
                    input_ids.size(),
                    ") must be >= embedding seq_len (",
                    seq_len,
                    "). Ensure input_ids are not from a stale or re-tokenized source.");

    auto* embed_data = input_embeds.data<float>();
    const auto* audio_data = m_audio_embeddings.data<const float>();
    const auto audio_total_tokens = m_audio_embeddings.get_shape()[0];
    const size_t bytes_per_token = hidden_size * sizeof(float);
    size_t audio_idx = 0;

    for (size_t i = 0; i < seq_len && audio_idx < audio_total_tokens; i++) {
        if (input_ids[i] == m_audio_token_id) {
            std::memcpy(embed_data + i * hidden_size, audio_data + audio_idx * hidden_size, bytes_per_token);
            audio_idx++;
        }
    }

    OPENVINO_ASSERT(audio_idx == audio_total_tokens,
                    "Audio token count mismatch: placed ",
                    audio_idx,
                    " embeddings but encoder produced ",
                    audio_total_tokens,
                    " tokens. Ensure the prompt contains exactly ",
                    audio_total_tokens,
                    " audio placeholder tokens.");
}

NormalizedPrompt InputsEmbedderQwen3Omni::normalize_prompt(const std::string& prompt,
                                                           size_t base_id,
                                                           const std::vector<EncodedImage>& images) const {
    auto result = normalize_prompt(prompt, base_id, 0, images, {});
    return {result.unified_prompt, result.images_sequence};
}

NormalizedPrompt InputsEmbedderQwen3Omni::normalize_prompt(const std::string& prompt,
                                                           size_t image_base_id,
                                                           size_t video_base_id,
                                                           const std::vector<EncodedImage>& images,
                                                           const std::vector<EncodedVideo>& videos) const {
    auto result = InputsEmbedderQwen3VL::normalize_prompt(prompt, image_base_id, video_base_id, images, videos);

    if (m_audio_embeddings && m_audio_embeddings.get_size() > 0) {
        const auto num_audio_tokens = m_audio_embeddings.get_shape()[0];

        const std::string audio_start = "<|audio_start|>";
        const std::string audio_pad = "<|audio_pad|>";
        const std::string audio_end = "<|audio_end|>";

        if (result.unified_prompt.find(audio_start) == std::string::npos) {
            std::string audio_tag;
            audio_tag.reserve(audio_start.size() + audio_pad.size() * num_audio_tokens + audio_end.size());
            audio_tag.append(audio_start);
            for (size_t i = 0; i < num_audio_tokens; ++i) {
                audio_tag.append(audio_pad);
            }
            audio_tag.append(audio_end);
            result.unified_prompt = audio_tag + result.unified_prompt;
        }
    }

    return result;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen3Omni::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence) {
    OPENVINO_ASSERT(has_merged_vision_model(),
                    "Merged vision model not loaded but images/videos were provided. "
                    "Ensure openvino_vision_embeddings_model.xml is present in the model directory.");

    auto [reordered_image_embeds, reordered_images_grid_thw] =
        qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] =
        qwen2_vl_utils::reorder_video_embeds_and_grid_thw(videos, videos_sequence);

    // These are raw patches now (not features) — shape [total_patches, patch_dim]
    auto concatenated_patches =
        qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.insert(combined_grid_thw.end(),
                             reordered_videos_grid_thw.begin(),
                             reordered_videos_grid_thw.end());
    combined_grid_thw.insert(combined_grid_thw.end(),
                             reordered_images_grid_thw.begin(),
                             reordered_images_grid_thw.end());

    // Compute pos_embeds as separate input (model adds them internally after Conv3d)
    ov::Tensor pos_embeds;
    if (!combined_grid_thw.empty()) {
        pos_embeds = get_interpolated_pos_embeds(combined_grid_thw);
    }

    auto rotary_pos_emb = get_rotary_pos_emb(combined_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_merged_vision.get());
    auto& ireq = guard.get();

    ireq.set_tensor("hidden_states", concatenated_patches);
    if (pos_embeds) {
        ireq.set_tensor("pos_embeds", pos_embeds);
    }
    // When VLSDPA fusion is active the GPU pass renamed "attention_mask" -> "cu_seq_lens" (packed
    // segment boundaries, i32). Otherwise feed the dense float attention_mask as before.
    if (m_with_cu_seqlens_input) {
        auto cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw);
        ireq.set_tensor("cu_seq_lens", cu_seq_lens);
    } else {
        auto attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw);
        ireq.set_tensor("attention_mask", attention_mask);
    }
    ireq.set_tensor("rotary_pos_emb", rotary_pos_emb);
    ireq.infer();

    auto vision_embeds = ireq.get_tensor("last_hidden_state");
    m_lm_extra_inputs["deepstack_visual_embeds"] = ireq.get_tensor("deepstack_feature_lists");

    const auto& vision_embeds_shape = vision_embeds.get_shape();

    const auto video_tokens = calc_vec_tokens_num(reordered_videos_grid_thw);
    const auto image_tokens = calc_vec_tokens_num(reordered_images_grid_thw);
    const auto total_tokens = video_tokens + image_tokens;

    size_t video_count = 0;
    if (total_tokens > 0) {
        video_count = vision_embeds_shape[0] * video_tokens / total_tokens;
    }
    const auto image_count = vision_embeds_shape[0] - video_count;

    ov::Tensor video_embeds{vision_embeds.get_element_type(), {video_count, vision_embeds_shape[1]}};
    ov::Tensor image_embeds{vision_embeds.get_element_type(), {image_count, vision_embeds_shape[1]}};

    std::memcpy(video_embeds.data(), vision_embeds.data(), video_embeds.get_byte_size());
    std::memcpy(image_embeds.data(),
                static_cast<uint8_t*>(vision_embeds.data()) + video_embeds.get_byte_size(),
                image_embeds.get_byte_size());

    return {video_embeds, image_embeds};
}

ov::Tensor InputsEmbedderQwen3Omni::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const {
    const auto spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        const auto t = grid_thw.at(0);
        const auto h = grid_thw.at(1);
        const auto w = grid_thw.at(2);

        max_grid_size = std::max({max_grid_size, h, w});

        std::vector<size_t> hpos_ids;
        std::vector<size_t> wpos_ids;
        const auto h_blocks = h / spatial_merge_size;
        const auto w_blocks = w / spatial_merge_size;
        hpos_ids.reserve(h * w);
        wpos_ids.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        hpos_ids.push_back(hb * spatial_merge_size + hs);
                        wpos_ids.push_back(wb * spatial_merge_size + ws);
                    }
                }
            }
        }

        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < hpos_ids.size(); ++j) {
                all_pos_ids.push_back({hpos_ids[j], wpos_ids[j]});
            }
        }
    }

    // Use cached rotary dim — Qwen3Omni uses a merged vision model instead of the separate
    // merger model that Qwen2VL/Qwen3VL use, so we cannot reuse the parent get_rotary_pos_emb()
    // which obtains dim via queue lock on m_ireq_queue_vision_embeddings_merger.
    OPENVINO_ASSERT(m_rotary_dim > 0, "Rotary dim not initialized — merged vision model not loaded");
    const auto dim = m_rotary_dim;
    const auto half_dim = dim / 2;
    constexpr float theta = 10000.0f;

    std::vector<float> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(half_dim));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(half_dim);
        for (size_t j = 0; j < half_dim; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    const size_t half_dim_bytes = half_dim * sizeof(float);
    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    auto* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids[i];
        const auto h_idx = pos[0];
        const auto w_idx = pos[1];
        std::memcpy(output_data + i * dim, freqs[h_idx].data(), half_dim_bytes);
        std::memcpy(output_data + i * dim + half_dim, freqs[w_idx].data(), half_dim_bytes);
    }

    return rotary_pos_emb;
}

void InputsEmbedderQwen3Omni::start_chat(const std::string& system_message) {
    InputsEmbedderQwen3VL::start_chat(system_message);
    m_audio_embeddings = ov::Tensor();
}

void InputsEmbedderQwen3Omni::finish_chat() {
    InputsEmbedderQwen3VL::finish_chat();
    m_audio_embeddings = ov::Tensor();
}

std::pair<ov::Tensor, int64_t> InputsEmbedderQwen3Omni::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t vision_start_token_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    // Create 4D position_ids for Qwen3-Omni's multimodal RoPE: [temporal, height, width, text]
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
    const size_t tokens_per_second = m_vlm_config.vision_config_tokens_per_second;

    auto reordered_vision_grid_thw = get_vision_grid_thw_for_position_ids(images_grid_thw,
                                                                          images_sequence,
                                                                          image_id,
                                                                          videos_grid_thw,
                                                                          videos_sequence,
                                                                          video_id,
                                                                          history_vision_count);

    const auto* input_ids = input_ids_tensor.data<int64_t>();
    size_t batch_size = input_ids_tensor.get_shape().at(0);
    size_t seq_len = input_ids_tensor.get_shape().at(1);

    OPENVINO_ASSERT(batch_size == 1, "create_position_ids currently supports only batch_size == 1");

    std::vector<size_t> vision_start_indices;
    for (size_t i = 0; i < seq_len; ++i) {
        if (input_ids[i] == vision_start_token_id) {
            vision_start_indices.push_back(i);
        }
    }

    // 4 dimensions: temporal, height, width, text
    ov::Tensor position_ids{ov::element::i64, {4, batch_size, seq_len}};
    auto* pos_data = position_ids.data<int64_t>();

    size_t st = 0;
    int64_t next_pos = 0;
    size_t grid_idx = 0;

    for (size_t i = 0; i < vision_start_indices.size(); ++i) {
        size_t ed = vision_start_indices.at(i);

        // Text tokens before vision: all 4 dims get the same sequential position
        if (st < ed) {
            for (size_t pos = st; pos < ed; ++pos) {
                pos_data[pos] = next_pos;                // temporal
                pos_data[seq_len + pos] = next_pos;      // height
                pos_data[2 * seq_len + pos] = next_pos;  // width
                pos_data[3 * seq_len + pos] = next_pos;  // text
                next_pos++;
            }
        }

        // Vision start token
        pos_data[ed] = next_pos;
        pos_data[seq_len + ed] = next_pos;
        pos_data[2 * seq_len + ed] = next_pos;
        pos_data[3 * seq_len + ed] = next_pos;
        next_pos++;
        ed++;

        // Vision tokens with spatial grid
        if (grid_idx < reordered_vision_grid_thw.size()) {
            const auto& grid = reordered_vision_grid_thw.at(grid_idx);
            size_t llm_grid_t = grid.at(0);
            size_t llm_grid_h = grid.at(1) / spatial_merge_size;
            size_t llm_grid_w = grid.at(2) / spatial_merge_size;
            size_t llm_grid_sz = llm_grid_h * llm_grid_w;
            size_t ed_image = ed + llm_grid_t * llm_grid_sz;

            // Temporal dimension
            for (size_t t = 0; t < llm_grid_t; t++) {
                std::fill_n(pos_data + ed + t * llm_grid_sz, llm_grid_sz, next_pos + t * tokens_per_second);
            }

            // Height and width dimensions
            int64_t* height_data = pos_data + seq_len + ed;
            int64_t* width_data = pos_data + 2 * seq_len + ed;
            for (size_t t = 0; t < llm_grid_t; t++) {
                size_t offset_sz = t * llm_grid_sz;
                for (size_t h = 0; h < llm_grid_h; ++h) {
                    size_t offset = h * llm_grid_w + offset_sz;
                    std::fill_n(height_data + offset, llm_grid_w, next_pos + h);
                    for (size_t w = 0; w < llm_grid_w; ++w) {
                        width_data[offset + w] = next_pos + w;
                    }
                }
            }

            // Text dimension: sequential position for all vision tokens
            int64_t* text_data = pos_data + 3 * seq_len + ed;
            for (size_t j = 0; j < llm_grid_t * llm_grid_sz; ++j) {
                text_data[j] = next_pos + static_cast<int64_t>(j);
            }

            next_pos += std::max(((llm_grid_t - 1) * tokens_per_second + 1), std::max(llm_grid_h, llm_grid_w));
            st = ed_image;
            grid_idx++;
        }
    }

    // Remaining text tokens
    if (st < seq_len) {
        for (size_t pos = st; pos < seq_len; ++pos) {
            pos_data[pos] = next_pos;
            pos_data[seq_len + pos] = next_pos;
            pos_data[2 * seq_len + pos] = next_pos;
            pos_data[3 * seq_len + pos] = next_pos;
            next_pos++;
        }
    }

    // Calculate rope delta from maximum position value 
    // (exclude text dimension which tracks sequence position but isn't consumed by RoPE)
    const size_t num_position_dims = position_ids.get_shape().at(0);
    const size_t num_spatial_dims = num_position_dims - 1;
    const int64_t position_ids_max_element = *std::max_element(
        position_ids.data<int64_t>(),
        position_ids.data<int64_t>() + num_spatial_dims * seq_len
    );
    const int64_t rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(seq_len);

    return {position_ids, rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen3Omni::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta) {
    OPENVINO_ASSERT(history_size != 0,
                    "get_generation_phase_position_ids() should only be called when history_size is non-zero.");
    // 4 dimensions for Qwen3-Omni's multimodal RoPE
    ov::Tensor position_ids{ov::element::i64, {4, 1, inputs_embeds_size}};
    int64_t new_pos_id = static_cast<int64_t>(history_size + rope_delta);
    for (size_t dim = 0; dim < 4; ++dim) {
        auto* pos_data = position_ids.data<int64_t>() + dim * inputs_embeds_size;
        std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
    }
    return {position_ids, rope_delta};
}

}  // namespace ov::genai
