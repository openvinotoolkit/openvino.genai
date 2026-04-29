
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/videochat_flash/classes.hpp"

#include "openvino/opsets/opset13.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "logger.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/vlm_utils.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <mutex>
#include <fstream>

// Forward-declare the rt_info helper from openvino's transformations library to avoid
// pulling in the private header "transformations/rt_info/disable_fp16_compression.hpp"
// (not on the genai include path). The symbol is exported via TRANSFORMATIONS_API and
// resolves at link time against the openvino runtime that genai already depends on.
namespace ov {
class Node;
void disable_fp16_compression(const std::shared_ptr<Node>& node);
} // namespace ov

namespace ov::genai {

namespace {

const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

// VideoChat-Flash still shares the top-level model config.json with the rest of the
// VLM stack, but these fields are intentionally parsed through a model-private schema.
// They are consumed only inside this implementation and are not part of the shared
// VLMConfig contract. Keeping them local avoids leaking VideoChat-Flash execution
// details into the generic VLM abstraction.
struct VideoChatFlashRuntimeConfig {
    size_t mm_local_num_frames = 4;
    size_t mm_hidden_size = 1408;
    size_t mm_num_attention_heads = 16;
    size_t mm_projector_num_tome_tokens = 64;
};

VideoChatFlashRuntimeConfig read_videochat_flash_runtime_config(const std::filesystem::path& config_path) {
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "' with VideoChat-Flash config");
    nlohmann::json parsed = nlohmann::json::parse(stream);

    VideoChatFlashRuntimeConfig config;
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "mm_local_num_frames", config.mm_local_num_frames);
    read_json_param(parsed, "mm_hidden_size", config.mm_hidden_size);
    read_json_param(parsed, "mm_num_attention_heads", config.mm_num_attention_heads);
    read_json_param(parsed, "mm_projector_num_tome_tokens", config.mm_projector_num_tome_tokens);
    return config;
}

void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>\n";
}

/**
 * @brief Build a frame-batch preprocessing model implemented purely with OpenVINO ops.
 *
 * Pipeline (all on device, batched, no per-frame CPU loop):
 *   1) Convert u8 -> f32
 *   2) Transpose NHWC -> NCHW
 *   3) Bicubic resize to (target_h, target_w) via Interpolate. Pinned to f32 with
 *      ov::disable_fp16_compression so fp16-compressing plugins (e.g. GPU) keep this
 *      single op in f32 -- see the inline comment near the Interpolate node for the
 *      numerical rationale. The other ops are linear/exact and stay at the device's
 *      default precision with no observable downstream impact.
 *   4) Clamp(0, 255) to mirror the value range of the legacy CPU uint8 staging buffer.
 *      Note: the integer round-trip (Add(0.5)+Floor) used by the legacy CPU reference
 *      path is intentionally not modelled here; with the bicubic pinned to f32 the
 *      sub-ULP residuals are absorbed by downstream LayerNorm/Softmax, matching the
 *      qwen2vl preprocess strategy.
 *   5) Normalize: (x / 255 - mean) / std, folded into
 *      (x - mean*255) * (1 / (std*255)) with broadcast constants.
 *
 * Inputs:
 *   - "raw_frames"    : u8  [N, H, W, 3] (dynamic N, H, W)
 *   - "resize_target" : i64 [2] = {target_h, target_w}
 * Output:
 *   - f32 [N, 3, target_h, target_w]
 *
 * The model is compiled once and reused across videos; the heavy per-frame
 * pixel work executes inside OpenVINO kernels instead of a C++ triple loop.
 */
std::shared_ptr<ov::Model> build_frame_preprocess_model(
    const std::array<float, 3>& image_mean,
    const std::array<float, 3>& image_std) {
    auto raw_frames = std::make_shared<ov::op::v0::Parameter>(
        ov::element::u8, ov::PartialShape{-1, -1, -1, 3});
    raw_frames->set_friendly_name("raw_frames");
    raw_frames->output(0).set_names({"raw_frames"});

    auto resize_target = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{2});
    resize_target->set_friendly_name("resize_target");
    resize_target->output(0).set_names({"resize_target"});

    // u8 NHWC -> f32 NCHW
    auto to_f32 = std::make_shared<ov::op::v0::Convert>(raw_frames, ov::element::f32);
    auto nhwc_to_nchw = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2});
    auto nchw = std::make_shared<ov::op::v1::Transpose>(to_f32, nhwc_to_nchw);

    // Bicubic resize on spatial dims (NCHW -> axes {2, 3})
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.cube_coeff = -0.75f;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;
    auto resize_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    auto resized = std::make_shared<ov::op::v11::Interpolate>(
        nchw, resize_target, resize_axes, attrs);
    // Pin only this op to f32 on fp16-compressing plugins (e.g. GPU). Default fp16
    // accumulation here makes the output snap to the fp16 grid (ULP ~0.0625 in the
    // 64..255 pixel range, e.g. 84.3125 = 84+5/16 instead of CPU's 84.3198), which
    // perturbs downstream ViT activations and shifts LLM tokens vs the CPU baseline.
    // The other preprocess ops (Convert/Transpose/Clamp/Subtract/Multiply) are linear
    // or exact and stay in fp16 with no observable downstream impact.
    ov::disable_fp16_compression(resized);

    auto clamped = std::make_shared<ov::op::v0::Clamp>(resized, 0.0, 255.0);

    // Fold (x/255 - mean)/std into (x - mean*255) * (1/(std*255)) with [1,3,1,1] broadcast consts.
    const std::vector<float> mean_scaled{
        image_mean[0] * 255.0f, image_mean[1] * 255.0f, image_mean[2] * 255.0f};
    const std::vector<float> inv_std_scaled{
        1.0f / (image_std[0] * 255.0f),
        1.0f / (image_std[1] * 255.0f),
        1.0f / (image_std[2] * 255.0f)};
    auto mean_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, 3, 1, 1}, mean_scaled);
    auto inv_std_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, 3, 1, 1}, inv_std_scaled);
    auto centered = std::make_shared<ov::op::v1::Subtract>(clamped, mean_const);
    auto normalized = std::make_shared<ov::op::v1::Multiply>(centered, inv_std_const);

    normalized->set_friendly_name("preprocessed_frames");
    return std::make_shared<ov::Model>(
        ov::OutputVector{normalized},
        ov::ParameterVector{raw_frames, resize_target},
        "videochat_flash_frame_preprocess");
}

/**
 * @brief Runtime selector for the frame preprocessing path.
 *
 *   default (env unset / != "CPP"): OV graph-based batched preprocess (Convert + Transpose +
 *                                   Bicubic + Normalize), compiled once and runs on the same
 *                                   device as the vision encoder.
 *   VISION_PREPROCESS=CPP         : Legacy CPU per-frame path (bicubic_resize +
 *                                   clip_image_preprocess). Kept for bit-accurate comparison
 *                                   and easy fallback during debugging / regression triage.
 */
bool check_vision_preprocess_env() {
    const char* env = std::getenv("VISION_PREPROCESS");
    return !(env && std::string(env) == "CPP");
}

/**
 * @brief Legacy CPU per-frame preprocess (kept for A/B comparison, selectable via env var).
 *
 * Pipeline:
 *   1) Resize each frame with bicubic interpolation to (target_h, target_w)
 *   2) Rescale pixel values by /255
 *   3) Normalize by (x - mean) / std
 *
 * Input:
 *   - input_nhwc_u8: [N, H, W, 3], element type u8
 * Output:
 *   - Tensor with shape [N, C, target_h, target_w], element type f32
 */
ov::Tensor preprocess_cpu_loop(const ov::Tensor& input_nhwc_u8,
                               ImageSize image_size,
                               const std::array<float, 3>& image_mean,
                               const std::array<float, 3>& image_std) {
    const ov::Shape& in_shape = input_nhwc_u8.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 4, "Input must be 4D NHWC.");
    OPENVINO_ASSERT(input_nhwc_u8.get_element_type() == ov::element::u8, "Input dtype must be u8.");
    OPENVINO_ASSERT(in_shape[3] == 3, "Input channel must be 3 for normalization.");
    OPENVINO_ASSERT(image_size.height > 0 && image_size.width > 0, "target_h and target_w must be > 0.");

    const size_t batch = in_shape[0];
    const size_t in_h = in_shape[1];
    const size_t in_w = in_shape[2];
    const size_t channels = in_shape[3];
    ov::Tensor output_nchw_f32(ov::element::f32, ov::Shape{batch, channels, image_size.height, image_size.width});
    float* out_ptr = output_nchw_f32.data<float>();

    const uint8_t* in_ptr = input_nhwc_u8.data<const uint8_t>();
    const size_t in_frame_bytes = in_h * in_w * channels;
    const size_t out_frame_elems = channels * image_size.height * image_size.width;

    clip_ctx ctx{};
    std::copy_n(image_mean.begin(), image_mean.size(), ctx.image_mean);
    std::copy_n(image_std.begin(), image_std.size(), ctx.image_std);

    for (size_t b = 0; b < batch; ++b) {
        ov::Tensor one_frame_u8(
            ov::element::u8,
            ov::Shape{1, in_h, in_w, channels},
            const_cast<uint8_t*>(in_ptr + b * in_frame_bytes)
        );

        // 1) resize (BICUBIC)
        clip_image_u8 clip_in = tensor_to_clip_image_u8(one_frame_u8);
        clip_image_u8 clip_resized;
        bicubic_resize(clip_in, clip_resized, image_size.width, image_size.height);

        // 2) rescale(/255) + 3) normalize((x-mean)/std), returns f32 planar CHW.
        clip_image_f32 clip_norm = clip_image_preprocess(ctx, clip_resized);

        OPENVINO_ASSERT(clip_norm.buf.size() == out_frame_elems, "Unexpected preprocessed frame size.");
        std::memcpy(out_ptr + b * out_frame_elems, clip_norm.buf.data(), out_frame_elems * sizeof(float));
    }

    return output_nchw_f32;
}

/**
 * @brief OV graph-based batched preprocess using a pre-compiled preprocessing model.
 *
 * Replaces the per-frame CPU pipeline with a single device-side inference over the whole
 * frame batch, eliminating N x (resize + normalize) CPU loops and the extra f32 planar->NCHW
 * copy.
 *
 * IMPORTANT lifetime contract (zero-copy optimization):
 *   The returned tensor is a NON-OWNING alias of the InferRequest's output buffer. The caller
 *   MUST keep the corresponding CircularBufferQueueElementGuard alive until the returned
 *   tensor's data has been fully consumed (typically until the next downstream copy, e.g.
 *   transpose_video_features). Reusing the same InferRequest before that point would corrupt
 *   the data. This avoids a ~70MB f32 copy_to per video chunk on the host side.
 *
 * Input:
 *   - input_nhwc_u8: [N, H, W, 3], element type u8
 * Output:
 *   - f32 tensor [N, 3, target_h, target_w] (alias to req.get_output_tensor())
 */
ov::Tensor preprocess_ov_graph(const ov::Tensor& input_nhwc_u8,
                               ImageSize image_size,
                               ov::InferRequest& req) {
    const ov::Shape& in_shape = input_nhwc_u8.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 4, "Input must be 4D NHWC.");
    OPENVINO_ASSERT(input_nhwc_u8.get_element_type() == ov::element::u8, "Input dtype must be u8.");
    OPENVINO_ASSERT(in_shape[3] == 3, "Input channel must be 3 for normalization.");
    OPENVINO_ASSERT(image_size.height > 0 && image_size.width > 0, "target_h and target_w must be > 0.");

    // Pack resize target as a 2-element i64 tensor.
    const std::vector<int64_t> target_shape_vec{
        static_cast<int64_t>(image_size.height),
        static_cast<int64_t>(image_size.width)};
    ov::Tensor target_shape_tensor(ov::element::i64, ov::Shape{2});
    std::memcpy(target_shape_tensor.data(), target_shape_vec.data(), target_shape_vec.size() * sizeof(int64_t));

    req.set_tensor("raw_frames", input_nhwc_u8);
    req.set_tensor("resize_target", target_shape_tensor);
    req.infer();

    // Return a non-owning alias of the request's output. Caller's guard keeps it valid;
    // the next downstream stage (transpose_video_features) memcpy's into its own buffer,
    // after which the guard can be released and the request returned to the pool.
    return req.get_output_tensor();
}

std::string normalize_prompt_impl(
    const std::string& prompt, size_t base_id, size_t n_visuals, const std::regex& native_pattern, void(*write_native)(std::ostream& os, size_t idx)
) {
    // Reject unsupported universal image placeholders early to keep the text+video contract explicit.
    std::smatch universal_image_match;
    std::regex_search(prompt, universal_image_match, UNIVERSAL_IMAGE_PATTERN);
    OPENVINO_ASSERT(
        universal_image_match.empty(),
        "VideoChat-Flash supports only text+video inputs now. "
        "Use <ov_genai_video_i> or native visual tags (<|image_i|>) for visuals."
    );

    // Convert universal video placeholders into the shared native visual tags.
    auto [normalized_prompt, visual_sequence] = universal_to_native(prompt, write_native, VisionType::VIDEO);
    if (!visual_sequence.empty()) {
        OPENVINO_ASSERT(
            !std::regex_search(prompt, native_pattern),
            "Prompt cannot mix universal video tags (<ov_genai_video_i>) with native visual tags (<|image_i|>)."
        );
        verify_ids(visual_sequence, base_id, n_visuals);
        return normalized_prompt;
    }

    // Preserve user-specified native tag ordering when the prompt already contains native visuals.
    for (std::sregex_iterator iter(prompt.begin(), prompt.end(), native_pattern), end;
         iter != end;
         ++iter) {
        size_t visual_id = std::stoul((*iter).str(1));
        OPENVINO_ASSERT(visual_id != 0, "Image tags must be greater than 0");
        visual_sequence.push_back(visual_id - 1);
    }
    if (!visual_sequence.empty()) {
        verify_ids(visual_sequence, base_id, n_visuals);
        return prompt;
    }

    // If the prompt has no visual placeholders, prepend them in the default input order.
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_visuals; relative_id++) {
        write_native(stream, base_id + relative_id);
    }
    stream << prompt;
    return stream.str();
}

ov::Tensor transpose_video_features(const ov::Tensor& src_tensor, const size_t mm_local_num_frames) {
    // Input feature:  [N, C, H, W]
    // Output feature with reshape & transpose: [N//mm_local_num_frames, C, mm_local_num_frames, H, W]
    const ov::Shape src_shape = src_tensor.get_shape();
    OPENVINO_ASSERT(src_shape.size() == 4, "Input tensor must be 4D [N, C, H, W].");
    OPENVINO_ASSERT(mm_local_num_frames > 0, "mm_local_num_frames must be greater than 0.");

    const size_t n = src_shape[0];
    const size_t c = src_shape[1];
    const size_t h = src_shape[2];
    const size_t w = src_shape[3];
    OPENVINO_ASSERT(n % mm_local_num_frames == 0,
                    "Batch size N must be divisible by mm_local_num_frames. Got N = ",
                    n, ", mm_local_num_frames = ", mm_local_num_frames, ".");
    const size_t n_prime = n / mm_local_num_frames;
    const ov::Shape dst_shape{n_prime, c, mm_local_num_frames, h, w};
    ov::Tensor dst_tensor(src_tensor.get_element_type(), dst_shape);

    const uint8_t* src_data = static_cast<const uint8_t*>(src_tensor.data());
    uint8_t* dst_data = static_cast<uint8_t*>(dst_tensor.data());
    const size_t elem_size = src_tensor.get_element_type().size();

    const size_t mchw = mm_local_num_frames * c * h * w;
    const size_t chw = c * h * w;
    const size_t hw = h * w;
    const size_t mhw = mm_local_num_frames * h * w;
    for (size_t np = 0; np < n_prime; ++np) {
        for (size_t cp = 0; cp < c; ++cp) {
            for (size_t dp = 0; dp < mm_local_num_frames; ++dp) {
                const size_t dst_idx = np * mchw + cp * mhw + dp * hw;
                const size_t src_idx = (np * mm_local_num_frames + dp) * chw + cp * hw;
                std::memcpy(
                    dst_data + dst_idx * elem_size,
                    src_data + src_idx * elem_size,
                    hw * elem_size);
            }
        }
    }

    return dst_tensor;
}

ov::Tensor remove_second_dim_first_element(const ov::Tensor& input) {
    const ov::Shape& input_shape = input.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 3, "Input tensor must be 3D [batch, seq, hidden], got ", input_shape.size(), "D.");
    OPENVINO_ASSERT(input_shape[1] >= 1, "Second dimension of input tensor must be at least 1.");

    const auto element_type = input.get_element_type();
    OPENVINO_ASSERT(element_type == ov::element::f32, "Input tensor element type must be f32.");

    const size_t element_size = element_type.size();
    OPENVINO_ASSERT(element_size > 0, "Unsupported tensor element type in remove_second_dim_first_element.");

    auto input_data = input.data<float>();
    ov::Shape output_shape = input_shape;
    const size_t org_seq_len = output_shape[1];
    output_shape[1] -= 1;
    const size_t seq_len = output_shape[1];
    const size_t head_elements = input_shape[2];
    ov::Tensor output(element_type, output_shape);
    auto output_data = output.data<float>();

    for(size_t i=0; i < input_shape[0]; i++) {
        std::copy(
            input_data + i * org_seq_len * head_elements + head_elements,
            input_data + (i + 1) * org_seq_len * head_elements,
            output_data + i * seq_len * head_elements
        );
    }

    return output;
}

std::shared_ptr<ov::Model> build_bipartite_soft_matching_merge_opt_model(int dim, ov::element::Type dtype = ov::element::f32) {
    // Parameters: x: [B, P, C], size: [B, P, 1]
    auto x_p = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape({-1, -1, -1}));
    x_p->set_friendly_name("hidden_states");
    x_p->output(0).set_names({"hidden_states"});
    auto size_p = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape({-1, -1, 1}));
    size_p->set_friendly_name("size");
    size_p->output(0).set_names({"size"});

    // Metric Calculation
    // metric4d = x.reshape(0, 0, -1, dim)
    auto reshape_pat = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, -1, dim});
    auto metric4d = std::make_shared<ov::op::v1::Reshape>(x_p, reshape_pat, true);

    // metric = reduce_mean(metric4d, axis=2) -> [B, P, dim]
    auto axis_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto metric = std::make_shared<ov::op::v1::ReduceMean>(metric4d, axis_2, false);

    // L2 Normalization
    // metric_n = metric / sqrt(sum(metric^2))
    auto metric_sq = std::make_shared<ov::op::v1::Multiply>(metric, metric);
    auto axis_neg1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto metric_ss = std::make_shared<ov::op::v1::ReduceSum>(metric_sq, axis_neg1, true);
    auto metric_norm = std::make_shared<ov::op::v0::Sqrt>(metric_ss);
    auto metric_n = std::make_shared<ov::op::v1::Divide>(metric, metric_norm);

    // Bipartite Indices (Even/Odd)
    auto shape_x = std::make_shared<ov::op::v3::ShapeOf>(x_p, ov::element::i64);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto p_node = std::make_shared<ov::op::v1::Gather>(shape_x,
                                                       std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}),
                                                       axis_0);

    // range(0, p, 2) and range(1, p, 2)
    auto const_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto const_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto const_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{2});

    auto idx_even = std::make_shared<ov::op::v4::Range>(const_0, p_node, const_2, ov::element::i64);
    auto idx_odd = std::make_shared<ov::op::v4::Range>(const_1, p_node, const_2, ov::element::i64);

    // Scoring & Matching
    auto axis_p = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto a = std::make_shared<ov::op::v1::Gather>(metric_n, idx_even, axis_p); // [B, P/2, dim]
    auto b = std::make_shared<ov::op::v1::Gather>(metric_n, idx_odd, axis_p);  // [B, P/2, dim]

    // scores = a @ b.T -> [B, P/2, P/2]
    auto b_t = std::make_shared<ov::op::v1::Transpose>(b, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1}));
    auto scores = std::make_shared<ov::op::v0::MatMul>(a, b_t, false, false);

    // TopK to get ArgMax (k=1)
    auto topk = std::make_shared<ov::op::v1::TopK>(scores,
                                                   std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}),
                                                   2, // axis
                                                   ov::op::v1::TopK::Mode::MAX,
                                                   ov::op::v1::TopK::SortType::SORT_VALUES,
                                                   ov::element::i64);
    auto node_idx = std::make_shared<ov::op::v0::Squeeze>(topk->output(1),
                                                          std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));

    // Merge Subgraph Logic
    auto merge_subgraph = [&](ov::Output<ov::Node> data_3d) {
        auto src = std::make_shared<ov::op::v1::Gather>(data_3d, idx_even, axis_p);
        auto dst = std::make_shared<ov::op::v1::Gather>(data_3d, idx_odd, axis_p);

        // Broadcast node_idx to [B, P/2, C]
        auto idx_u = std::make_shared<ov::op::v0::Unsqueeze>(node_idx,
                                                             std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));
        auto src_shape = std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64);
        auto idx_b = std::make_shared<ov::op::v1::Broadcast>(idx_u, src_shape);

        auto merged = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            dst, idx_b, src, const_1, ov::op::v12::ScatterElementsUpdate::Reduction::SUM
        );
        return merged;
    };

    // Final Weighted Merge
    auto x_weighted = std::make_shared<ov::op::v1::Multiply>(x_p, size_p);
    auto x_m = merge_subgraph(x_weighted);
    auto size_m = merge_subgraph(size_p);
    auto x_out = std::make_shared<ov::op::v1::Divide>(x_m, size_m);

    // Construct Model
    x_out->set_friendly_name("x_out");
    size_m->set_friendly_name("size_out");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{x_out, size_m}, ov::ParameterVector{x_p, size_p}, "bipartite_merge_opt");
    return model;
}

// Merge visual tokens with iterative ToMe until reaching ``target_num_token``.
// Each merge round removes at most half of current tokens due to bipartite matching constraints.
ov::Tensor merge_tokens(const ov::Tensor& input, ov::InferRequest& merge_embeddings, const size_t target_num_token = 64) {
    const ov::Shape& x_shape = input.get_shape();
    OPENVINO_ASSERT(
        x_shape.size() == 3,
        "x must be 3D tensor [batch, tokens, channels], got ", x_shape.size(), "D."
    );

    const size_t b = x_shape[0];
    const size_t p = x_shape[1];
    const size_t c = x_shape[2];

    OPENVINO_ASSERT(
        target_num_token > 0,
        "target_num_token must be greater than 0. Got ", target_num_token, "."
    );
    OPENVINO_ASSERT(
        p >= target_num_token,
        "Current tokens (", p, ") must be greater than or equal to target (", target_num_token, ")."
    );

    if (p == target_num_token) {
        return input;
    }

    size_t reachable_tokens = p;
    while (reachable_tokens > target_num_token) {
        OPENVINO_ASSERT(
            reachable_tokens % 2 == 0,
            "Token count must stay even during bipartite merge. Current=", reachable_tokens, "."
        );
        reachable_tokens /= 2;
    }
    OPENVINO_ASSERT(
        reachable_tokens == target_num_token,
        "target_num_token is not reachable by repeated halving: initial=", p,
        ", target=", target_num_token, "."
    );

    const ov::Shape size_shape = {b, p, 1};
    ov::Tensor size_tensor(ov::element::f32, size_shape);
    float* size_data = size_tensor.data<float>();
    const size_t num_elements = size_tensor.get_size();
    std::fill(size_data, size_data + num_elements, 1.0f);

    ov::Tensor current_x = input;

    while (true) {
        const ov::Shape& current_shape = current_x.get_shape();
        OPENVINO_ASSERT(
            current_shape.size() == 3 && current_shape[0] == b && current_shape[2] == c,
            "hidden_states must keep shape [", b, ", tokens, ", c, "], got ",
            current_shape.to_string(), "."
        );

        const size_t current_tokens = current_shape[1];
        if (current_tokens == target_num_token) {
            break;
        }

        OPENVINO_ASSERT(
            current_tokens > target_num_token,
            "Unexpected token count before merge: ", current_tokens,
            ", target=", target_num_token, "."
        );

        const ov::Shape& current_size_shape = size_tensor.get_shape();
        OPENVINO_ASSERT(
            current_size_shape.size() == 3 &&
                current_size_shape[0] == b &&
                current_size_shape[1] == current_tokens &&
                current_size_shape[2] == 1,
            "size tensor must match hidden_states shape [", b, ", ", current_tokens, ", 1], got ",
            current_size_shape.to_string(), "."
        );

        merge_embeddings.set_tensor("hidden_states", current_x);
        merge_embeddings.set_tensor("size", size_tensor);
        merge_embeddings.infer();

        ov::Tensor next_x = merge_embeddings.get_output_tensor(0);
        ov::Tensor next_size = merge_embeddings.get_output_tensor(1);

        const ov::Shape& next_x_shape = next_x.get_shape();
        const ov::Shape& next_size_shape = next_size.get_shape();

        OPENVINO_ASSERT(
            next_x_shape.size() == 3 && next_x_shape[0] == b && next_x_shape[2] == c,
            "merge output hidden_states has invalid shape: ", next_x_shape.to_string(), "."
        );
        OPENVINO_ASSERT(
            next_size_shape.size() == 3 &&
                next_size_shape[0] == b &&
                next_size_shape[1] == next_x_shape[1] &&
                next_size_shape[2] == 1,
            "merge output size has invalid shape: ", next_size_shape.to_string(),
            ", expected [", b, ", ", next_x_shape[1], ", 1]."
        );
        OPENVINO_ASSERT(
            next_x_shape[1] == current_tokens / 2,
            "merge model must halve token count each step. Current=", current_tokens,
            ", next=", next_x_shape[1], "."
        );

        current_x = std::move(next_x);
        size_tensor = std::move(next_size);
    }

    const ov::Shape expected_shape = {b, target_num_token, c};
    OPENVINO_ASSERT(
        current_x.get_shape() == expected_shape,
        "Merge failed: expected shape ", expected_shape.to_string(),
        ", got ", current_x.get_shape().to_string()
    );

    return current_x;
}

ov::Tensor efficient_flatten(const ov::Tensor& original_tensor) {
    // flatten 3D tensor [N,C,W] to 3D tensor [1, N*C, W]
    const ov::Shape& original_shape = original_tensor.get_shape();
    OPENVINO_ASSERT(
        original_shape.size() == 3,
        "efficient_flatten expects a 3D tensor with layout [N, C, W], got ",
        original_shape.size(),
        "D."
    );
    const ov::element::Type& dtype = original_tensor.get_element_type();
    const ov::Shape new_shape = {
        1,
        original_shape[0] * original_shape[1], // N * C
        original_shape[2]                      // W
    };
    ov::Tensor new_tensor(dtype, new_shape);
    OPENVINO_ASSERT(
        original_tensor.get_size() == new_tensor.get_size(),
        "Flatten error: Element count mismatch during reshape."
    );
    const void* src_data = original_tensor.data();
    void* dst_data = new_tensor.data();
    std::memcpy(dst_data, src_data, original_tensor.get_byte_size());
    return new_tensor;
}
std::vector<float> get_1d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<float>& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0, "embed_dim must be divisible by 2 for 1D sin-cos positional embedding.");
    int M = pos.size();
    int half_dim = embed_dim / 2;

    std::vector<float> omega(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, (float)i / (embed_dim / 2.0f));
    }

    std::vector<float> emb(M * embed_dim);
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < half_dim; ++d) {
            float out = pos[m] * omega[d];
            emb[m * embed_dim + d] = std::sin(out);            // First half uses sine
            emb[m * embed_dim + d + half_dim] = std::cos(out); // Second half uses cosine
        }
    }
    return emb;
}

std::vector<float> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<float>>& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0, "embed_dim must be divisible by 2 for 2D sin-cos positional embedding.");
    int half_dim = embed_dim / 2;
    int num_points = grid[0].size();

    auto emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0]);
    auto emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1]);

    std::vector<float> emb(num_points * embed_dim);
    for (int i = 0; i < num_points; ++i) {
        std::copy(emb_h.begin() + i * half_dim, emb_h.begin() + (i + 1) * half_dim, emb.begin() + i * embed_dim);
        std::copy(emb_w.begin() + i * half_dim, emb_w.begin() + (i + 1) * half_dim, emb.begin() + i * embed_dim + half_dim);
    }
    return emb;
}

ov::Tensor get_3d_sincos_pos_embed(int embed_dim, int grid_size, int t_size, bool cls_token = false) {
    OPENVINO_ASSERT(embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 3D sin-cos positional embedding.");
    int embed_dim_spatial = (embed_dim / 4) * 3;
    int embed_dim_temporal = embed_dim / 4;
    int hw = grid_size * grid_size;
    int total_tokens = t_size * hw;
    int num_rows = (cls_token ? 1 : 0) + total_tokens;

    ov::Shape output_shape = {1, static_cast<size_t>(num_rows), static_cast<size_t>(embed_dim)};
    ov::Tensor pos_tensor(ov::element::f32, output_shape);

    float* out_ptr = pos_tensor.data<float>();
    std::fill(out_ptr, out_ptr + pos_tensor.get_size(), 0.0f);

    std::vector<std::vector<float>> grid(2, std::vector<float>(hw));
    for (int h = 0; h < grid_size; ++h) {
        for (int w = 0; w < grid_size; ++w) {
            grid[0][h * grid_size + w] = (float)w;
            grid[1][h * grid_size + w] = (float)h;
        }
    }
    auto pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid);

    std::vector<float> grid_t(t_size);
    std::iota(grid_t.begin(), grid_t.end(), 0.0f);
    auto pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t);

    int output_offset = cls_token ? 1 : 0;
    for (int t = 0; t < t_size; ++t) {
        for (int i = 0; i < hw; ++i) {
            float* token_ptr = out_ptr + (output_offset + t * hw + i) * embed_dim;
            std::memcpy(token_ptr, 
                        &pos_embed_temporal[t * embed_dim_temporal], 
                        embed_dim_temporal * sizeof(float));
            std::memcpy(token_ptr + embed_dim_temporal, 
                        &pos_embed_spatial[i * embed_dim_spatial], 
                        embed_dim_spatial * sizeof(float));
        }
    }

    return pos_tensor;
}
/**
 * @brief Run the ViT (vision_embeddings) on N temporally-grouped video samples and return a
 *        single contiguous [N, ...] tensor.
 *
 * Per-sample zero-copy loop. Dry-infer sample 0 to learn per-sample output shape, then
 * pre-allocate the [N, ...] result buffer and feed every subsequent sample's output slice
 * as the request's output tensor via set_output_tensor() so the plugin writes directly into
 * the final buffer (no per-sample memcpy). The output binding is reset to a scratch tensor
 * before returning so the request can be safely returned to the pool.
 *
 * Pre-conditions on the InferRequest (set up in initialize_vision_encoder_queue's factory):
 *   - "rotary_pos_emb" is already bound to m_pos_emb on every request in the pool.
 *
 * Note: a true-batch path (single infer with hidden_states = [N, C, T, H, W]) was evaluated
 * but regressed performance on Arc 140V iGPU at typical N — the per-sample loop hits better
 * GPU kernel tiles and stays within the device working set, so it is the only path kept.
 */
ov::Tensor cyclic_vit_infer(const ov::Tensor& transpose_features,
                            ov::InferRequest& vision_embeddings) {
    OPENVINO_ASSERT(
        transpose_features.get_element_type() == ov::element::f32,
        "vision_embeddings input pixel_values must be f32."
    );

    const ov::Shape full_shape = transpose_features.get_shape();
    OPENVINO_ASSERT(full_shape.size() == 5, "transpose_features must be 5D [N, C, T, H, W].");
    const size_t N = full_shape[0];
    OPENVINO_ASSERT(N > 0, "transpose_features batch size N must be greater than 0.");

    const size_t single_sample_size = transpose_features.get_size() / N;
    const float* src_ptr = transpose_features.data<const float>();
    const ov::Shape single_in_shape = {1, full_shape[1], full_shape[2], full_shape[3], full_shape[4]};

    // Dry-infer sample 0 to learn per-sample output shape.
    ov::Tensor first_in(ov::element::f32, single_in_shape, const_cast<float*>(src_ptr));
    vision_embeddings.set_tensor("hidden_states", first_in);
    vision_embeddings.infer();
    ov::Tensor first_out = vision_embeddings.get_output_tensor();

    const ov::Shape per_out_shape = first_out.get_shape();
    const ov::element::Type out_dtype = first_out.get_element_type();
    OPENVINO_ASSERT(
        !per_out_shape.empty() && per_out_shape[0] == 1,
        "vision_embeddings per-sample output is expected to have batch dim == 1, got shape ",
        per_out_shape.to_string(), "."
    );

    ov::Shape final_shape = per_out_shape;
    final_shape[0] = N;
    ov::Tensor final_processed_embeds(out_dtype, final_shape);
    const size_t per_sample_bytes = first_out.get_byte_size();
    OPENVINO_ASSERT(
        final_processed_embeds.get_byte_size() == per_sample_bytes * N,
        "Final ViT output buffer size mismatch."
    );
    uint8_t* dst_base = static_cast<uint8_t*>(final_processed_embeds.data());
    std::memcpy(dst_base, first_out.data(), per_sample_bytes);

    bool foreign_output_bound = false;
    for (size_t i = 1; i < N; ++i) {
        ov::Tensor view_in(
            ov::element::f32, single_in_shape,
            const_cast<float*>(src_ptr + i * single_sample_size));
        ov::Tensor view_out(
            out_dtype, per_out_shape,
            dst_base + i * per_sample_bytes);
        vision_embeddings.set_tensor("hidden_states", view_in);
        vision_embeddings.set_output_tensor(view_out);
        vision_embeddings.infer();
        foreign_output_bound = true;
    }
    if (foreign_output_bound) {
        // Rebind output to a scratch tensor so the InferRequest, once returned to the pool,
        // does not write into our just-returned final_processed_embeds buffer.
        ov::Tensor scratch(out_dtype, per_out_shape);
        vision_embeddings.set_output_tensor(scratch);
    }

    return final_processed_embeds;
}

} // namespace

void VisionEncoderVideoChatFlashQwen::initialize_positional_embedding() {
    const size_t mm_hidden_size = m_mm_hidden_size;
    const size_t mm_local_num_frames = m_mm_local_num_frames;
    OPENVINO_ASSERT(m_processor_config.image_size > 0, "image_size must be greater than 0.");
    OPENVINO_ASSERT(m_processor_config.patch_size > 0, "patch_size must be greater than 0.");
    OPENVINO_ASSERT(
        m_processor_config.image_size % m_processor_config.patch_size == 0,
        "image_size must be divisible by patch_size. Got image_size=",
        m_processor_config.image_size,
        ", patch_size=",
        m_processor_config.patch_size,
        "."
    );
    const size_t grid_size = m_processor_config.image_size / m_processor_config.patch_size;
    m_pos_emb = get_3d_sincos_pos_embed(mm_hidden_size, grid_size, mm_local_num_frames, true);
    OPENVINO_ASSERT(
        m_pos_emb.get_size() > 0,
        "m_pos_emb must be initialized before vision embeddings model initialization."
    );
}

void VisionEncoderVideoChatFlashQwen::initialize_runtime_config(const std::filesystem::path& config_path) {
    const auto runtime_config = read_videochat_flash_runtime_config(config_path);
    OPENVINO_ASSERT(runtime_config.mm_local_num_frames > 0, "mm_local_num_frames must be greater than 0.");
    OPENVINO_ASSERT(runtime_config.mm_hidden_size > 0, "mm_hidden_size must be greater than 0.");
    OPENVINO_ASSERT(runtime_config.mm_num_attention_heads > 0, "mm_num_attention_heads must be greater than 0.");
    OPENVINO_ASSERT(runtime_config.mm_projector_num_tome_tokens > 0, "mm_projector_num_tome_tokens must be greater than 0.");

    m_mm_local_num_frames = runtime_config.mm_local_num_frames;
    m_mm_hidden_size = runtime_config.mm_hidden_size;
    m_num_attention_heads = runtime_config.mm_num_attention_heads;
    m_target_num_token = runtime_config.mm_projector_num_tome_tokens;
}

void VisionEncoderVideoChatFlashQwen::initialize_preprocess_queue(const std::string& device, const ov::AnyMap& properties) {
    // Build the preprocess model (Convert + Transpose + Bicubic resize + Clamp + Normalize)
    // and compile it on the caller-requested device. The bicubic Interpolate inside the
    // graph is tagged with ov::disable_fp16_compression so the GPU plugin keeps just that
    // single op in f32 (see comment in build_frame_preprocess_model); the rest of the
    // graph still runs at the device's default precision and the caller's `properties`
    // are forwarded unchanged.
    auto preprocess_model = build_frame_preprocess_model(
        m_processor_config.image_mean, m_processor_config.image_std);
    auto compiled_preprocess = utils::singleton_core().compile_model(preprocess_model, device, properties);
    m_ireq_queue_preprocess = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_preprocess.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_preprocess]() -> ov::InferRequest {
            return compiled_preprocess.create_infer_request();
        });
}

void VisionEncoderVideoChatFlashQwen::initialize_merge_model_queue() {
    const size_t num_attention_heads = m_num_attention_heads;
    OPENVINO_ASSERT(
        m_mm_hidden_size % num_attention_heads == 0,
        "mm_hidden_size must be divisible by num_attention_heads for VideoChat-Flash merge model. Got mm_hidden_size=",
        m_mm_hidden_size, ", num_attention_heads=", num_attention_heads, "."
    );

    const size_t merge_dim = m_mm_hidden_size / num_attention_heads;
    auto merge_model = build_bipartite_soft_matching_merge_opt_model(merge_dim);
    // TODO: CVS-183390 This pipeline is typically used with a GPU;
    // however, when this model runs on the GPU as well,
    // the GPU memory footprint does not meet the target requirements, so we must use the CPU for now.
    auto compiled_merge_model = utils::singleton_core().compile_model(merge_model, "CPU", {});
    m_ireq_queue_merge_model = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_merge_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_merge_model]() -> ov::InferRequest {
            return compiled_merge_model.create_infer_request();
        });
}

void VisionEncoderVideoChatFlashQwen::initialize_shared_config(const std::filesystem::path& config_dir_path) {
    // VideoChat-Flash exports do not provide preprocessor_config.json.
    // The required preprocessing parameters for this model live in config.json,
    // so ProcessorConfig is initialized from that file explicitly.
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, "config.json");
    m_video_processor_config = utils::from_config_json_if_exists<VideoProcessorConfig>(config_dir_path, "video_preprocessor_config.json");
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    initialize_runtime_config(config_dir_path / "config.json");
}

void VisionEncoderVideoChatFlashQwen::initialize_vision_encoder_queue(
    std::shared_ptr<ov::Model> model,
    const std::string& device,
    const ov::AnyMap& properties) {
    initialize_positional_embedding();
    const ov::Shape pos_emb_shape = m_pos_emb.get_shape();

    OPENVINO_ASSERT(
        pos_emb_shape.size() == 3 && pos_emb_shape[0] == 1,
        "m_pos_emb must have shape [1, tokens, hidden]. Got ",
        pos_emb_shape.to_string(),
        "."
    );

    // Accelerate model by using static rope shape.
    std::map<std::string, ov::PartialShape> input_shapes;
    input_shapes["rotary_pos_emb"] = pos_emb_shape;  // it's a fixed shape: { 1, 1025, 1408 }
    model->reshape(input_shapes);

    auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    // rotary_pos_emb is constant across the entire pipeline lifetime (depends only on grid_size
    // and mm_local_num_frames). Bind it once at request-construction time so cyclic_vit_infer
    // does not re-set it on every per-sample call. Its lifetime is managed by InferRequest
    // after set_tensor().
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model, this]() -> ov::InferRequest {
            auto req = compiled_model.create_infer_request();
            req.set_tensor("rotary_pos_emb", m_pos_emb);
            return req;
        });
}

void VisionEncoderVideoChatFlashQwen::initialize_vision_projection_queue(ov::CompiledModel& compiled_model) {
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

VisionEncoderVideoChatFlashQwen::VisionEncoderVideoChatFlashQwen(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) {
    initialize_shared_config(model_dir);
    m_use_ov_vision_preprocess = check_vision_preprocess_env();

    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
    initialize_vision_encoder_queue(model, device, properties);

    auto compiled_model_vision =
        utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, properties);
    initialize_vision_projection_queue(compiled_model_vision);

    initialize_merge_model_queue();
    if (m_use_ov_vision_preprocess) {
        initialize_preprocess_queue(device, properties);
    }
}

VisionEncoderVideoChatFlashQwen::VisionEncoderVideoChatFlashQwen(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) {
    initialize_shared_config(config_dir_path);
    m_use_ov_vision_preprocess = check_vision_preprocess_env();

    const auto& [vision_embeddings_model, vision_embeddings_weights] =
        utils::get_model_weights_pair(models_map, "vision_embeddings");
    auto model = utils::singleton_core().read_model(vision_embeddings_model, vision_embeddings_weights);
    initialize_vision_encoder_queue(model, device, properties);

    const auto& [vision_projection_model, vision_projection_weights] =
        utils::get_model_weights_pair(models_map, "vision_projection");
    auto compiled_model = utils::singleton_core().compile_model(vision_projection_model, vision_projection_weights, device, properties);
    initialize_vision_projection_queue(compiled_model);

    initialize_merge_model_queue();
    if (m_use_ov_vision_preprocess) {
        initialize_preprocess_queue(device, properties);
    }
}

EncodedImage VisionEncoderVideoChatFlashQwen::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    (void)image;
    (void)config_map;
    OPENVINO_THROW("VideoChat-Flash currently does not support image inference. Please use video input.");
}

VisionEncoderVideoChatFlashQwen::PreprocessFunc VisionEncoderVideoChatFlashQwen::get_preprocess_func() {
    if (m_use_ov_vision_preprocess) {
        return [this](const ov::Tensor& sampled_video,
                      ImageSize target_size,
                      std::optional<CircularBufferQueueElementGuard<ov::InferRequest>>& preprocess_guard) -> ov::Tensor {
            // OV-graph preprocess returns a non-owning alias of the InferRequest output
            // buffer; the caller must keep `preprocess_guard` alive until the next downstream
            // copy (transpose_video_features) consumes it.
            preprocess_guard.emplace(this->m_ireq_queue_preprocess.get());
            return preprocess_ov_graph(sampled_video, target_size, preprocess_guard->get());
        };
    } else {
        return [this](const ov::Tensor& sampled_video,
                      ImageSize target_size,
                      std::optional<CircularBufferQueueElementGuard<ov::InferRequest>>& /*preprocess_guard*/) -> ov::Tensor {
            // Legacy CPU per-frame loop owns its output buffer; no guard required.
            return preprocess_cpu_loop(
                sampled_video, target_size,
                this->m_processor_config.image_mean, this->m_processor_config.image_std);
        };
    }
}

ov::Tensor VisionEncoderVideoChatFlashQwen::sample_video_if_needed(const ov::Tensor& video) const {
    const size_t frames_group_size = m_mm_local_num_frames;
    OPENVINO_ASSERT(frames_group_size > 0, "mm_local_num_frames must be greater than 0.");
    const ov::Shape& video_shape = video.get_shape();
    OPENVINO_ASSERT(video_shape.size() == 4, "Input video tensor must be 4D [N, H, W, C].");

    const size_t frame_count = video_shape[0];
    OPENVINO_ASSERT(frame_count > 0, "Input video frame_count must be greater than 0.");

    const size_t remainder = frame_count % frames_group_size;
    if (remainder == 0) {
        return video;
    }

    // The strategy of video sampling in original model is complex, here we just simply pad the video by replicating the
    // last frame until the frame count is divisible by frames_group_size.
    // Reference: https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B/blob/main/mm_utils.py#L107
    // TODO: CVS-183520 to implement the original sampling strategy.
    const size_t pad_frame_count = frames_group_size - remainder;
    const size_t padded_frame_count = frame_count + pad_frame_count;
    GENAI_WARN("Video frame_count=%zu is not divisible by group_size=%zu. Padding %zu frame(s), padded_frame_count=%zu.",
               frame_count,
               frames_group_size,
               pad_frame_count,
               padded_frame_count);

    ov::Shape padded_shape = video_shape;
    padded_shape[0] = padded_frame_count;
    ov::Tensor padded_video(video.get_element_type(), padded_shape);

    const size_t element_size = video.get_element_type().size();
    OPENVINO_ASSERT(element_size > 0, "Unsupported tensor element type in sample_video_if_needed.");

    const size_t frame_volume = video_shape[1] * video_shape[2] * video_shape[3];
    const size_t frame_bytes = frame_volume * element_size;
    const size_t original_bytes = frame_count * frame_bytes;
    const uint8_t* src_data = static_cast<const uint8_t*>(video.data());
    uint8_t* dst_data = static_cast<uint8_t*>(padded_video.data());
    std::memcpy(dst_data, src_data, original_bytes);

    // Replicate the last valid frame to fill the required group size.
    const uint8_t* last_frame = src_data + (frame_count - 1) * frame_bytes;
    for (size_t i = 0; i < pad_frame_count; ++i) {
        std::memcpy(dst_data + (frame_count + i) * frame_bytes, last_frame, frame_bytes);
    }

    return padded_video;
}

EncodedVideo VisionEncoderVideoChatFlashQwen::encode_video(const ov::Tensor& video) {
    EncodedVideo encoded_video;
    OPENVINO_ASSERT(m_processor_config.image_size > 0, "image_size must be greater than 0.");
    ImageSize target_size{m_processor_config.image_size, m_processor_config.image_size};

    ov::Tensor sampled_video = sample_video_if_needed(video);

    // The OV-graph preprocess returns a non-owning alias of its InferRequest output buffer.
    // The guard below keeps that buffer alive across transpose_video_features (which performs
    // the only required copy into its own tensor), and is then released so the request can
    // return to the pool. The legacy CPU path doesn't need the guard.
    std::optional<CircularBufferQueueElementGuard<ov::InferRequest>> preprocess_guard;
    auto preprocess_func = get_preprocess_func();
    ov::Tensor preprocessed_video = preprocess_func(sampled_video, target_size, preprocess_guard);

    encoded_video.resized_source_size = target_size;
    encoded_video.frame_num = preprocessed_video.get_shape()[0];
    const size_t mm_local_num_frames = m_mm_local_num_frames;
    auto transpose_features = transpose_video_features(preprocessed_video, mm_local_num_frames);
    // transpose_video_features has copied all preprocess output into its own buffer; the
    // preprocess InferRequest can be safely returned to the pool now.
    preprocess_guard.reset();

    CircularBufferQueueElementGuard<ov::InferRequest> vision_guard(m_ireq_queue_vision_encoder.get());
    CircularBufferQueueElementGuard<ov::InferRequest> merge_guard(m_ireq_queue_merge_model.get());
    CircularBufferQueueElementGuard<ov::InferRequest> projection_guard(m_ireq_queue_vision_projection.get());

    ov::Tensor processed_vision_embeds = cyclic_vit_infer(transpose_features, vision_guard.get());
    ov::Tensor clipped_vision_embeds = remove_second_dim_first_element(processed_vision_embeds);
    ov::Tensor merged_vision_features = merge_tokens(clipped_vision_embeds, merge_guard.get(), m_target_num_token);

    projection_guard.get().set_input_tensor(merged_vision_features);
    projection_guard.get().infer();
    ov::Tensor proj_features = projection_guard.get().get_output_tensor();

    ov::Tensor final_features = efficient_flatten(proj_features);

    encoded_video.video_features = final_features;
    encoded_video.num_video_tokens = final_features.get_shape()[1];
    return encoded_video;
}



std::vector<ov::genai::EncodedVideo> InputsEmbedderVideoChatFlashQwen::encode_videos(const std::vector<ov::Tensor>& videos) {
    const auto vision_encoder = std::static_pointer_cast<VisionEncoderVideoChatFlashQwen>(m_vision_encoder);
    std::vector<EncodedVideo> embeds;
    for (const ov::Tensor& video : videos) {
        embeds.emplace_back(vision_encoder->encode_video(video));
    }
    return embeds;
}

InputsEmbedderVideoChatFlashQwen::InputsEmbedderVideoChatFlashQwen(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderVideoChatFlashQwen::InputsEmbedderVideoChatFlashQwen(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}


NormalizedPrompt InputsEmbedderVideoChatFlashQwen::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {normalize_prompt_impl(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
}

NormalizedPrompt InputsEmbedderVideoChatFlashQwen::normalize_prompt(
    const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos) const {
    OPENVINO_ASSERT(
        images.empty(),
        "VideoChat-Flash does not support image inputs. Please provide video inputs only."
    );

    const size_t base_visual_id = std::max(base_video_id, base_image_id + images.size());
    const size_t total_visuals = images.size() + videos.size();
    return {normalize_prompt_impl(prompt, base_visual_id, total_visuals, NATIVE_PATTERN, write_native), {}};
}

ov::Tensor InputsEmbedderVideoChatFlashQwen::get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    size_t base_id = m_tokens_per_images.size();
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }
    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = vlm_utils::split_tokenize(prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = prompt;
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = vlm_utils::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    ov::Tensor new_merged_tokens = vlm_utils::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_cache_state.get_state().size();
    m_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = vlm_utils::drop_image_placeholders(new_tokens);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = vlm_utils::build_inputs_embeds_from_text_and_visual_chunks(
        tokens,
        [&](const ov::Tensor& text_chunk) {
            return m_embedding->infer(req, text_chunk);
        },
        images_features_proj,
        base_id,
        new_tokens.get_shape().at(1),
        m_vlm_config.hidden_size
    );

    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }
    return inputs_embeds;
}

ov::Tensor InputsEmbedderVideoChatFlashQwen::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    if (!videos_sequence.empty()) {
         static std::once_flag videos_sequence_warning_once;
         std::call_once(videos_sequence_warning_once, [&videos_sequence]() {
             GENAI_WARN("VideoChat-Flash ignores separate video sequence in this path. videos_sequence size=%zu.", videos_sequence.size());
         });
    }

    std::vector<ov::genai::EncodedImage> combined_images = images;
    combined_images.reserve(images.size() + videos.size());
    for (const auto& video : videos) {
        ov::genai::EncodedImage as_image;
        as_image.images_features_projection = video.video_features;
        combined_images.emplace_back(std::move(as_image));
    }

    return get_inputs_embeds(prompt, combined_images, metrics, recalculate_merged_embeddings, image_sequence);
}

void InputsEmbedderVideoChatFlashQwen::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL)
        m_tokens_per_images = m_prev_tokens_per_images;
    else
        m_prev_tokens_per_images = m_tokens_per_images;
}

void InputsEmbedderVideoChatFlashQwen::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
    m_prev_tokens_per_images.clear();
}

void InputsEmbedderVideoChatFlashQwen::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
    m_prev_tokens_per_images.clear();
}
} // namespace ov::genai
