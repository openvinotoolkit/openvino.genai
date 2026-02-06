// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_vae_decoder_tiling.hpp"

#include "module_genai/module_factory.hpp"

#include <fstream>
#include <cstring>
#include <chrono>
#include <cmath>

#include "json_utils.hpp"
#include "module_genai/utils/tensor_utils.hpp"
#include "module_genai/utils/blend_utils.hpp"
#include "module_genai/utils/profiler.hpp"
#include "utils.hpp"

#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(VAEDecoderTilingModule);

void VAEDecoderTilingModule::print_static_config() {
    std::cout << R"(
  vae_decoder_tiling:
    type: "VAEDecoderTilingModule"
    model_type: "zimage" or "wan2.1"           # Determines 4D (image) or 5D (video) processing
    device: "CPU"
    inputs:
      - name: "latent"
        type: "OVTensor"                       # 4D [N,C,H,W] or 5D [B,C,T,H,W]
        source: "ParentModuleName.OutputPortName"
      - name: "latents"
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "image"                          # For 4D image output [N,H,W,C] u8
        type: "OVTensor"
      - name: "video"                          # For 5D video output [B,T,H,W,C] u8
        type: "OVTensor"
      - name: "images"
        type: "VecOVTensor"
      - name: "videos"
        type: "VecOVTensor"
    params:
      tile_overlap_factor: "0.25"              # [Optional] float, default is 0.25
      sample_size: "1024"                      # [Optional] int, tiling size for images
      tile_sample_min_height: "256"            # [Optional] tile size for videos
      tile_sample_min_width: "256"
      tile_sample_stride: "192"                # [Optional] stride for videos
      spatial_compression_ratio: "8"           # [Optional] VAE spatial downsampling
      model_path: "model"
      sub_module_name: "sub_modules: name"     # sub-pipeline module name

    )" << std::endl;
}

VAEDecoderTilingModule::VAEDecoderTilingModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    m_model_type = to_diffusion_model_type(desc->model_type);

    // Determine content type based on model type
    if (is_video_generation_model(m_model_type)) {
        m_content_type = ContentType::VIDEO;
        GENAI_INFO("VAEDecoderTilingModule: Video mode (5D tensors) for model type: " + desc->model_type);
    } else if (is_image_generation_model(m_model_type)) {
        m_content_type = ContentType::IMAGE;
        GENAI_INFO("VAEDecoderTilingModule: Image mode (4D tensors) for model type: " + desc->model_type);
    } else {
        GENAI_ERR("VAEDecoderTilingModule[" + desc->name + "]: Unsupported model type: " + desc->model_type);
        return;
    }

    if (!initialize()) {
        GENAI_ERR("VAEDecoderTilingModule: Failed to initialize");
    }
}

VAEDecoderTilingModule::~VAEDecoderTilingModule() {}

bool VAEDecoderTilingModule::init_tile_params_from_config() {
    // Read tiling parameters from YAML config (for video mode)
    auto tile_h = get_optional_param("tile_sample_min_height");
    if (!tile_h.empty()) {
        m_tile_sample_min_size = std::stoi(tile_h);
    }

    auto tile_w = get_optional_param("tile_sample_min_width");
    if (!tile_w.empty() && tile_h.empty()) {
        // Use height for both if width not specified separately; fall back to width when height is not set
        m_tile_sample_min_size = std::stoi(tile_w);
    }

    auto stride = get_optional_param("tile_sample_stride");
    if (!stride.empty()) {
        m_tile_sample_stride = std::stoi(stride);
    } else {
        // Calculate stride from overlap factor
        int overlap = static_cast<int>(m_tile_sample_min_size * m_tile_overlap_factor);
        m_tile_sample_stride = m_tile_sample_min_size - overlap;
    }

    auto ratio = get_optional_param("spatial_compression_ratio");
    if (!ratio.empty()) {
        m_spatial_compression_ratio = std::stoi(ratio);
    }

    // Calculate latent tile size
    m_tile_latent_min_size = m_tile_sample_min_size / m_spatial_compression_ratio;

    int overlap = m_tile_sample_min_size - m_tile_sample_stride;
    GENAI_INFO("VAEDecoderTilingModule: Tiling config (video mode):");
    GENAI_INFO("  - Tile size: " + std::to_string(m_tile_sample_min_size) + " pixels");
    GENAI_INFO("  - Stride: " + std::to_string(m_tile_sample_stride) + " pixels");
    GENAI_INFO("  - Overlap: " + std::to_string(overlap) + " pixels");
    GENAI_INFO("  - Latent tile: " + std::to_string(m_tile_latent_min_size));
    GENAI_INFO("  - Compression ratio: " + std::to_string(m_spatial_compression_ratio) + "x");

    return true;
}

bool VAEDecoderTilingModule::init_tile_params(const std::filesystem::path& model_path) {
    auto factor = get_optional_param("tile_overlap_factor");
    if (!factor.empty()) {
        m_tile_overlap_factor = std::stof(factor);
    }

    bool have_sample_size = false;
    auto sample_size = get_optional_param("sample_size");
    if (!sample_size.empty()) {
        m_sample_size = std::stoi(sample_size);
        m_tile_sample_min_size = m_sample_size;
        have_sample_size = true;
    }


    auto config_path = model_path / "vae_decoder/config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream vae_config(config_path);
        nlohmann::json parsed = nlohmann::json::parse(vae_config);
        if (!have_sample_size) {
            ov::genai::utils::read_json_param(parsed, "sample_size", m_sample_size);
            m_tile_sample_min_size = m_sample_size;
        }

        // get block_out_channels: "block_out_channels": [128,256,512,512]
        std::vector<int> block_out_channels;
        ov::genai::utils::read_json_param(parsed, "block_out_channels", block_out_channels);

        // m_tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        m_tile_latent_min_size = m_sample_size / std::pow(2, block_out_channels.size() - 1);
    } else {
        OPENVINO_ASSERT(false,
                        "VAEDecoderTilingModule[" + module_desc->name + "]: vae_decoder config file not found at " +
                            config_path.string());
    }

    return true;
}

bool VAEDecoderTilingModule::init_post_process() {
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;

    // Construct post-processing OV model here.
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1, -1});
    auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
    auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
    auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(input, constant_0_5);
    auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
    auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
    auto multiplied = std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
    auto result = std::make_shared<ov::op::v0::Result>(multiplied);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.output().postprocess().convert_element_type(ov::element::u8);
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");
    ppp.build();

    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, device);
    pp_infer_request = compiled_model.create_infer_request();

    return true;
}

bool VAEDecoderTilingModule::initialize() {
    const auto& params = module_desc->params;

    // Read overlap factor first (used by both modes)
    auto factor = get_optional_param("tile_overlap_factor");
    if (!factor.empty()) {
        m_tile_overlap_factor = std::stof(factor);
    }

    // Initialize tiling parameters based on content type
    if (m_content_type == ContentType::IMAGE) {
        // Image mode: read from model config
        auto it_path = params.find("model_path");
        if (it_path == params.end()) {
            GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: 'model_path' not found in params");
            return false;
        }
        std::filesystem::path model_path = module_desc->get_full_path(it_path->second);
        init_tile_params(model_path);
    } else {
        // Video mode: read from YAML config
        init_tile_params_from_config();
    }

    auto it_sub_module_name = module_desc->params.find("sub_module_name");
    if (it_sub_module_name == params.end()) {
        GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: 'sub_module_name' not found in params");
        return false;
    }

    m_sub_pipeline_impl = init_sub_pipeline(it_sub_module_name->second, pipeline_desc, module_desc);
    if (!m_sub_pipeline_impl) {
        return false;
    }

    // Post-processing only needed for image mode (video mode handles it in sub-pipeline)
    if (m_content_type == ContentType::IMAGE) {
        if (!init_post_process()) {
            return false;
        }
        m_slice_infer_request = tensor_utils::init_slice_request(module_desc->device.empty() ? "CPU" : module_desc->device);
    }

    GENAI_INFO("VAEDecoderTilingModule: Initialized successfully");
    return true;
}

void VAEDecoderTilingModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);
    prepare_inputs();

    std::vector<ov::Tensor> latents;
    if (this->inputs.find("latent") != this->inputs.end()) {
        latents.push_back(this->inputs["latent"].data.as<ov::Tensor>());
    } else if (this->inputs.find("latents") != this->inputs.end()) {
        if (this->inputs["latents"].data.is<ov::Tensor>()) {
            latents.push_back(this->inputs["latents"].data.as<ov::Tensor>());
        } else {
            latents = this->inputs["latents"].data.as<std::vector<ov::Tensor>>();
        }
    } else {
        GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: 'latent' or 'latents' input not found");
        return;
    }

    std::vector<ov::Tensor> outputs;

    for (const auto& latent : latents) {
        ov::Tensor output;

        if (m_content_type == ContentType::IMAGE) {
            // ========== 4D Image Processing ==========
            ov::Tensor cur_latent = latent;
            if (latent.get_shape().size() == 3u) {
                cur_latent = ov::Tensor(latent.get_element_type(),
                                        ov::Shape{1, latent.get_shape()[0], latent.get_shape()[1], latent.get_shape()[2]},
                                        latent.data());
            }
            OPENVINO_ASSERT(cur_latent.get_shape().size() == 4u,
                            "VAEDecoderTilingModule: Image mode expects 4D tensor. Got: " +
                            tensor_utils::shape_to_string(cur_latent.get_shape()));

            size_t height = cur_latent.get_shape()[2];
            size_t width = cur_latent.get_shape()[3];

            if (m_enable_tiling &&
                (height > static_cast<size_t>(m_tile_latent_min_size) ||
                 width > static_cast<size_t>(m_tile_latent_min_size))) {
                tile_decode_4d(cur_latent, output);
            } else {
                GENAI_INFO("VAEDecoderTilingModule: Direct decode (4D), size [" +
                           std::to_string(height) + "x" + std::to_string(width) + "]");
                output = decoder(cur_latent);
            }

            // Post-process for images
            pp_infer_request.set_input_tensor(output);
            {
                PROFILE(pm, "post-process infer");
                pp_infer_request.infer();
            }
            output = pp_infer_request.get_output_tensor();
            ov::Tensor pp_out = ov::Tensor(output.get_element_type(), output.get_shape());
            output.copy_to(pp_out);
            output = pp_out;

        } else {
            // ========== 5D Video Processing ==========
            OPENVINO_ASSERT(latent.get_shape().size() == 5u,
                            "VAEDecoderTilingModule: Video mode expects 5D tensor [B,C,T,H,W]. Got: " +
                            tensor_utils::shape_to_string(latent.get_shape()));

            size_t height = latent.get_shape()[3];
            size_t width = latent.get_shape()[4];

            if (m_enable_tiling &&
                (height > static_cast<size_t>(m_tile_latent_min_size) ||
                 width > static_cast<size_t>(m_tile_latent_min_size))) {
                tile_decode_5d(latent, output);
            } else {
                GENAI_INFO("VAEDecoderTilingModule: Direct decode (5D), size [" +
                           std::to_string(height) + "x" + std::to_string(width) + "]");
                output = decoder(latent);
            }
            // No post-process needed for video (handled in sub-pipeline)
        }

        outputs.push_back(output);
    }

    // Set outputs based on content type
    if (m_content_type == ContentType::IMAGE) {
        if (outputs.size() == 1) {
            this->outputs["image"].data = outputs[0];
        } else {
            this->outputs["images"].data = outputs;
        }
    } else {
        if (outputs.size() == 1) {
            this->outputs["video"].data = outputs[0];
        } else {
            this->outputs["videos"].data = outputs;
        }
    }
}

ov::Tensor VAEDecoderTilingModule::decoder(const ov::Tensor& tile) {
    ov::AnyMap inputs;

    // Use appropriate input name based on content type
    if (m_content_type == ContentType::VIDEO) {
        inputs["latent"] = tile;
    } else {
        inputs["latents"] = tile;
    }

    m_sub_pipeline_impl->generate(inputs);

    // Try multiple output names
    std::vector<std::string> output_names = {"video", "image"};
    for (const auto& name : output_names) {
        ov::Any output = m_sub_pipeline_impl->get_output(name);
        if (!output.empty() && output.is<ov::Tensor>()) {
            auto output_tensor = output.as<ov::Tensor>();
            ov::Tensor clone_tensor(output_tensor.get_element_type(), output_tensor.get_shape());
            output_tensor.copy_to(clone_tensor);
            return clone_tensor;
        }
    }

    GENAI_ERR("VAEDecoderTilingModule[" + module_desc->name + "]: Sub-pipeline output not found");
    return ov::Tensor();
}

void VAEDecoderTilingModule::tile_decode_4d(const ov::Tensor& latent, ov::Tensor& output_latent) {
    // Tiling decode implementation
    size_t overlap_size = m_tile_latent_min_size * (1 - m_tile_overlap_factor);
    size_t blend_extent = m_tile_sample_min_size * m_tile_overlap_factor;
    size_t row_limit = m_tile_sample_min_size - blend_extent;

    size_t height = latent.get_shape()[2];
    size_t width = latent.get_shape()[3];

    std::vector<std::vector<ov::Tensor>> rows;
    for (size_t h = 0; h < height; h += overlap_size) {
        const size_t h_start = h;
        const size_t h_end = std::min(h + m_tile_latent_min_size, height);
        std::vector<ov::Tensor> row;
        for (size_t w = 0; w < width; w += overlap_size) {
            const size_t w_start = w;
            const size_t w_end = std::min(w + m_tile_latent_min_size, width);

            // Get ROI tile from latent
            ov::Tensor tile = ov::genai::module::tensor_utils::slice_tensor(
                latent,
                {0, 0, h_start, w_start},
                {latent.get_shape()[0], latent.get_shape()[1], h_end, w_end});

            ov::Tensor decoded_tile = decoder(tile);

            row.push_back(decoded_tile);
        }
        rows.push_back(row);
    }

    std::vector<ov::Tensor> result_rows;
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<ov::Tensor> result_row;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            ov::Tensor tile = rows[i][j];
            if (i > 0) {
                blend_utils::blend_v_4d(rows[i - 1][j], tile, blend_extent);
            }
            if (j > 0) {
                blend_utils::blend_h_4d(rows[i][j - 1], tile, blend_extent);
            }
            const auto dst_shape = tile.get_shape();
            result_row.push_back(tensor_utils::slice_tensor(
                tile,
                {0, 0, 0, 0},
                {dst_shape[0], dst_shape[1], std::min(dst_shape[2], row_limit), std::min(dst_shape[3], row_limit)}));
        }
        result_rows.push_back(tensor_utils::concat_tensors(result_row, 3));
    }

    // Combine result_rows into output_latent
    output_latent = tensor_utils::concat_tensors(result_rows, 2);
}

// ============================================================================
// 5D Tensor Processing (Video)
// ============================================================================

ov::Tensor VAEDecoderTilingModule::slice_5d(const ov::Tensor& tensor,
                                             size_t h_start, size_t h_end,
                                             size_t w_start, size_t w_end) {
    auto shape = tensor.get_shape();
    size_t B = shape[0], C = shape[1], T = shape[2], H = shape[3], W = shape[4];

    size_t tile_h = h_end - h_start;
    size_t tile_w = w_end - w_start;

    ov::Tensor tile(tensor.get_element_type(), {B, C, T, tile_h, tile_w});

    const float* src = tensor.data<float>();
    float* dst = tile.data<float>();

    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t h = 0; h < tile_h; ++h) {
                    size_t src_offset = ((b * C + c) * T + t) * H * W + (h_start + h) * W + w_start;
                    size_t dst_offset = ((b * C + c) * T + t) * tile_h * tile_w + h * tile_w;
                    std::memcpy(dst + dst_offset, src + src_offset, tile_w * sizeof(float));
                }
            }
        }
    }

    return tile;
}

ov::Tensor VAEDecoderTilingModule::pad_tile_5d(const ov::Tensor& tile, size_t target_h, size_t target_w) {
    auto shape = tile.get_shape();
    size_t B = shape[0], C = shape[1], T = shape[2], H = shape[3], W = shape[4];

    if (H >= target_h && W >= target_w) {
        return tile;
    }

    ov::Tensor padded(tile.get_element_type(), {B, C, T, target_h, target_w});

    const float* src = tile.data<float>();
    float* dst = padded.data<float>();
    std::memset(dst, 0, padded.get_byte_size());

    // Reflection padding
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t h = 0; h < target_h; ++h) {
                    for (size_t w = 0; w < target_w; ++w) {
                        size_t src_h = h < H ? h : (2 * H - h - 2);
                        size_t src_w = w < W ? w : (2 * W - w - 2);
                        src_h = std::min(src_h, H - 1);
                        src_w = std::min(src_w, W - 1);

                        size_t src_idx = ((b * C + c) * T + t) * H * W + src_h * W + src_w;
                        size_t dst_idx = ((b * C + c) * T + t) * target_h * target_w + h * target_w + w;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }

    return padded;
}

ov::Tensor VAEDecoderTilingModule::crop_decoded_tile_5d(const ov::Tensor& decoded,
                                                         size_t orig_h, size_t orig_w,
                                                         size_t full_h, size_t full_w) {
    auto shape = decoded.get_shape();
    size_t B = shape[0], T = shape[1], H = shape[2], W = shape[3], C = shape[4];

    size_t target_h = orig_h * m_spatial_compression_ratio;
    size_t target_w = orig_w * m_spatial_compression_ratio;

    if (H == target_h && W == target_w) {
        return decoded;
    }

    ov::Tensor cropped(decoded.get_element_type(), {B, T, target_h, target_w, C});

    const uint8_t* src = decoded.data<uint8_t>();
    uint8_t* dst = cropped.data<uint8_t>();

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t h = 0; h < target_h; ++h) {
                size_t src_idx = ((b * T + t) * H + h) * W * C;
                size_t dst_idx = ((b * T + t) * target_h + h) * target_w * C;
                std::memcpy(dst + dst_idx, src + src_idx, target_w * C * sizeof(uint8_t));
            }
        }
    }

    return cropped;
}

ov::Tensor VAEDecoderTilingModule::concat_tiles_5d(const std::vector<std::vector<ov::Tensor>>& tiles,
                                                    size_t stride_h, size_t stride_w) {
    if (tiles.empty() || tiles[0].empty()) {
        OPENVINO_THROW("VAEDecoderTilingModule: Empty tiles for concatenation");
    }

    auto first_shape = tiles[0][0].get_shape();
    bool is_u8 = (tiles[0][0].get_element_type() == ov::element::u8);

    size_t num_rows = tiles.size();
    size_t num_cols = tiles[0].size();

    size_t B, T, C, total_h, total_w;

    if (is_u8) {
        // [B, T, H, W, C]
        B = first_shape[0];
        T = first_shape[1];
        C = first_shape[4];
        total_h = (num_rows - 1) * stride_h + tiles.back()[0].get_shape()[2];
        total_w = (num_cols - 1) * stride_w + tiles[0].back().get_shape()[3];
    } else {
        // [B, C, T, H, W]
        B = first_shape[0];
        C = first_shape[1];
        T = first_shape[2];
        total_h = (num_rows - 1) * stride_h + tiles.back()[0].get_shape()[3];
        total_w = (num_cols - 1) * stride_w + tiles[0].back().get_shape()[4];
    }

    ov::Shape result_shape = is_u8 ? ov::Shape{B, T, total_h, total_w, C}
                                    : ov::Shape{B, C, T, total_h, total_w};
    ov::Tensor result(tiles[0][0].get_element_type(), result_shape);

    size_t h_offset = 0;
    for (size_t row = 0; row < num_rows; ++row) {
        size_t w_offset = 0;
        size_t tile_h = is_u8 ? tiles[row][0].get_shape()[2] : tiles[row][0].get_shape()[3];
        size_t copy_h = (row == num_rows - 1) ? tile_h : stride_h;

        for (size_t col = 0; col < num_cols; ++col) {
            const auto& tile = tiles[row][col];
            auto tile_shape = tile.get_shape();
            size_t tile_w = is_u8 ? tile_shape[3] : tile_shape[4];
            size_t copy_w = (col == num_cols - 1) ? tile_w : stride_w;

            if (is_u8) {
                const uint8_t* src = tile.data<uint8_t>();
                uint8_t* dst = result.data<uint8_t>();

                for (size_t b = 0; b < B; ++b) {
                    for (size_t t = 0; t < T; ++t) {
                        for (size_t h = 0; h < copy_h; ++h) {
                            size_t src_idx = ((b * T + t) * tile_shape[2] + h) * tile_shape[3] * C;
                            size_t dst_idx = ((b * T + t) * total_h + h_offset + h) * total_w * C + w_offset * C;
                            std::memcpy(dst + dst_idx, src + src_idx, copy_w * C * sizeof(uint8_t));
                        }
                    }
                }
            } else {
                const float* src = tile.data<float>();
                float* dst = result.data<float>();

                for (size_t b = 0; b < B; ++b) {
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t t = 0; t < T; ++t) {
                            for (size_t h = 0; h < copy_h; ++h) {
                                size_t src_idx = ((b * C + c) * T + t) * tile_shape[3] * tile_shape[4] + h * tile_shape[4];
                                size_t dst_idx = ((b * C + c) * T + t) * total_h * total_w + (h_offset + h) * total_w + w_offset;
                                std::memcpy(dst + dst_idx, src + src_idx, copy_w * sizeof(float));
                            }
                        }
                    }
                }
            }

            w_offset += stride_w;
        }
        h_offset += stride_h;
    }

    return result;
}

void VAEDecoderTilingModule::tile_decode_5d(const ov::Tensor& latent, ov::Tensor& output) {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto shape = latent.get_shape();
    size_t batch = shape[0], channels = shape[1], num_frames = shape[2];
    size_t height = shape[3];
    size_t width = shape[4];

    int tile_latent_h = m_tile_sample_min_size / m_spatial_compression_ratio;
    int tile_latent_w = m_tile_sample_min_size / m_spatial_compression_ratio;
    int stride_latent_h = m_tile_sample_stride / m_spatial_compression_ratio;
    int stride_latent_w = m_tile_sample_stride / m_spatial_compression_ratio;

    size_t blend_h = m_tile_sample_min_size - m_tile_sample_stride;
    size_t blend_w = m_tile_sample_min_size - m_tile_sample_stride;

    size_t num_rows = (height + stride_latent_h - 1) / stride_latent_h;
    size_t num_cols = (width + stride_latent_w - 1) / stride_latent_w;
    size_t total_tiles = num_rows * num_cols;

    GENAI_INFO("========== VAE Tiled Decode (5D Video) ==========");
    GENAI_INFO("Input: [" + std::to_string(batch) + ", " + std::to_string(channels) +
               ", " + std::to_string(num_frames) + ", " + std::to_string(height) + ", " + std::to_string(width) + "]");
    GENAI_INFO("Tile grid: " + std::to_string(num_rows) + "x" + std::to_string(num_cols) +
               " = " + std::to_string(total_tiles) + " tiles");

    std::vector<std::vector<ov::Tensor>> rows;
    size_t tile_idx = 0;
    double total_decode_ms = 0.0;

    for (size_t i = 0; i < height; i += stride_latent_h) {
        std::vector<ov::Tensor> row;
        for (size_t j = 0; j < width; j += stride_latent_w) {
            size_t h_end = std::min(i + static_cast<size_t>(tile_latent_h), height);
            size_t w_end = std::min(j + static_cast<size_t>(tile_latent_w), width);

            size_t orig_h = h_end - i;
            size_t orig_w = w_end - j;
            bool needs_padding = (orig_h < static_cast<size_t>(tile_latent_h) ||
                                  orig_w < static_cast<size_t>(tile_latent_w));

            ov::Tensor tile = slice_5d(latent, i, h_end, j, w_end);
            if (needs_padding) {
                tile = pad_tile_5d(tile, tile_latent_h, tile_latent_w);
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            ov::Tensor decoded = decoder(tile);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_decode_ms += ms;

            if (needs_padding) {
                decoded = crop_decoded_tile_5d(decoded, orig_h, orig_w, tile_latent_h, tile_latent_w);
            }

            tile_idx++;
            std::string pad_info = needs_padding ? " (padded)" : "";
            GENAI_INFO("  Tile " + std::to_string(tile_idx) + "/" + std::to_string(total_tiles) +
                       pad_info + " - " + std::to_string(static_cast<int>(ms)) + " ms");

            row.push_back(decoded);
        }
        rows.push_back(row);
    }

    // Blend
    auto blend_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < rows[i].size(); ++j) {
            if (i > 0) {
                blend_utils::blend_v_5d(rows[i - 1][j], rows[i][j], blend_h);
            }
            if (j > 0) {
                blend_utils::blend_h_5d(rows[i][j - 1], rows[i][j], blend_w);
            }
        }
    }
    auto blend_end = std::chrono::high_resolution_clock::now();
    double blend_ms = std::chrono::duration<double, std::milli>(blend_end - blend_start).count();

    // Concat
    auto concat_start = std::chrono::high_resolution_clock::now();
    output = concat_tiles_5d(rows, m_tile_sample_stride, m_tile_sample_stride);
    auto concat_end = std::chrono::high_resolution_clock::now();
    double concat_ms = std::chrono::duration<double, std::milli>(concat_end - concat_start).count();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    GENAI_INFO("  Decode: " + std::to_string(static_cast<int>(total_decode_ms)) + " ms");
    GENAI_INFO("  Blend:  " + std::to_string(static_cast<int>(blend_ms)) + " ms");
    GENAI_INFO("  Concat: " + std::to_string(static_cast<int>(concat_ms)) + " ms");
    GENAI_INFO("  Total:  " + std::to_string(static_cast<int>(total_ms)) + " ms");
    GENAI_INFO("Output: [" + std::to_string(output.get_shape()[0]) + ", " +
               std::to_string(output.get_shape()[1]) + ", " +
               std::to_string(output.get_shape()[2]) + ", " +
               std::to_string(output.get_shape()[3]) + ", " +
               std::to_string(output.get_shape()[4]) + "]");
    GENAI_INFO("=================================================");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
