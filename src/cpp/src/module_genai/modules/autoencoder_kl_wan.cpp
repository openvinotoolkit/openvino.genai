// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "autoencoder_kl_wan.hpp"
#include "utils.hpp"
#include "json_utils.hpp"
#include "logger.hpp"
#include "module_genai/utils/blend_utils.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/runtime/properties.hpp"
#include "module_genai/utils/profiler.hpp"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace ov::genai::module {

AutoencoderKLWan::Config::Config(const std::filesystem::path &config_path) {
    if (!std::filesystem::exists(config_path)) {
        OPENVINO_THROW("AutoencoderKLWan config file does not exist: " + config_path.string());
    }
    std::ifstream config_file(config_path);
    nlohmann::json parsed = nlohmann::json::parse(config_file);
    utils::read_json_param(parsed, "base_dim", base_dim);
    utils::read_json_param(parsed, "z_dim", z_dim);
    utils::read_json_param(parsed, "num_res_blocks", num_res_blocks);
    utils::read_json_param(parsed, "dim_mult", dim_mult);
    utils::read_json_param(parsed, "dropout", dropout);
    utils::read_json_param(parsed, "latents_mean", latents_mean);
    utils::read_json_param(parsed, "latents_std", latents_std);
    utils::read_json_param(parsed, "temperal_downsample", temperal_downsample);
}

AutoencoderKLWan::AutoencoderKLWan(const std::filesystem::path &vae_decoder_path,
                                   const std::string &device,
                                   const ov::AnyMap &properties)
    : m_config(vae_decoder_path / "config.json") {
    if (std::filesystem::exists(vae_decoder_path / "openvino_model.xml")) {
        m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    } else if (std::filesystem::exists(vae_decoder_path / "vae_decoder.xml")) {
        m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "vae_decoder.xml");
    } else {
        OPENVINO_THROW("AutoencoderKLWan decoder model file does not exist in: " + vae_decoder_path.string());
    }

    auto properties_copy = properties;
    m_enable_postprocess = true;
    if (auto it = properties_copy.find("enable_postprocess"); it != properties_copy.end()) {
        m_enable_postprocess = it->second.as<bool>();
        properties_copy.erase(it);
    }

    // Check for warmup option (default: enabled)
    bool do_warmup = true;
    if (auto it = properties_copy.find("warmup"); it != properties_copy.end()) {
        do_warmup = it->second.as<bool>();
        properties_copy.erase(it);
    }

    // Setup model cache directory (speeds up subsequent runs)
    // User can override with their own CACHE_DIR in properties
    if (properties_copy.find(ov::cache_dir.name()) == properties_copy.end()) {
        // Use a default cache directory next to the model
        std::filesystem::path cache_dir = vae_decoder_path / "ov_cache";
        std::filesystem::create_directories(cache_dir);
        properties_copy[ov::cache_dir.name()] = cache_dir.string();
        GENAI_INFO("AutoencoderKLWan: Using model cache: " + cache_dir.string());
    }

    init_prepostprocess(m_enable_postprocess);

    auto compile_start = std::chrono::high_resolution_clock::now();
    m_decoder_request = utils::singleton_core().compile_model(m_decoder_model, device, properties_copy).create_infer_request();
    auto compile_end = std::chrono::high_resolution_clock::now();
    double compile_time_ms = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
    GENAI_INFO("AutoencoderKLWan: Model compiled in " + std::to_string(static_cast<int>(compile_time_ms)) + " ms");

    // Automatic warmup if enabled
    if (do_warmup) {
        warmup();
    }
}

void AutoencoderKLWan::init_prepostprocess(bool enable_postprocess) {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);
    ppp.input().tensor().set_layout("NCDHW");
    ppp.output().model().set_layout("NCDHW");
    std::vector<float> inv_std, neg_mean;
    for (size_t i = 0; i < m_config.latents_mean.size(); i++) {
        inv_std.push_back(1.0f / m_config.latents_std[i]);
        neg_mean.push_back(-m_config.latents_mean[i]);
    }

    ppp.input().preprocess()
        .scale(inv_std)
        .mean(neg_mean);

    if (enable_postprocess) {
        ppp.output().postprocess().custom([](const ov::Output<ov::Node> &port) {
            auto permute = ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{5},
                {0, 2, 1, 3, 4});
            auto transposed = std::make_shared<ov::op::v1::Transpose>(port, permute);
            auto constant_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
            auto scaled_0_5 = std::make_shared<ov::op::v1::Multiply>(transposed, constant_0_5);
            auto added_0_5 = std::make_shared<ov::op::v1::Add>(scaled_0_5, constant_0_5);
            auto clamped = std::make_shared<ov::op::v0::Clamp>(added_0_5, 0.0f, 1.0f);
            auto constant_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);
            auto multiplied = std::make_shared<ov::op::v1::Multiply>(clamped, constant_255);
            auto permute_1 = ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{5},
                {0, 1, 3, 4, 2});
            return std::make_shared<ov::op::v1::Transpose>(multiplied, permute_1);
        });
        ppp.output().postprocess().convert_element_type(ov::element::u8);
    }
    m_decoder_model = ppp.build();
}

void AutoencoderKLWan::warmup(size_t num_frames) {
    if (m_warmed_up) {
        GENAI_INFO("AutoencoderKLWan: Model already warmed up, skipping");
        return;
    }

    // Create a dummy latent tensor with the fixed tile size
    // Shape: [B=1, C=16, T=num_frames, H=tile_latent_h, W=tile_latent_w]
    int tile_latent_h = m_tiling_config.tile_sample_min_height / m_tiling_config.spatial_compression_ratio;
    int tile_latent_w = m_tiling_config.tile_sample_min_width / m_tiling_config.spatial_compression_ratio;

    ov::Shape dummy_shape = {1, 16, num_frames,
                             static_cast<size_t>(tile_latent_h),
                             static_cast<size_t>(tile_latent_w)};

    GENAI_INFO("AutoencoderKLWan: Warming up model with dummy input [1, 16, " +
               std::to_string(num_frames) + ", " + std::to_string(tile_latent_h) +
               ", " + std::to_string(tile_latent_w) + "]...");

    auto warmup_start = std::chrono::high_resolution_clock::now();

    // Create dummy tensor filled with zeros
    ov::Tensor dummy(ov::element::f32, dummy_shape);
    std::memset(dummy.data<float>(), 0, dummy.get_byte_size());

    // Run inference to trigger JIT compilation
    m_decoder_request.set_input_tensor(dummy);
    m_decoder_request.infer();

    auto warmup_end = std::chrono::high_resolution_clock::now();
    double warmup_time_ms = std::chrono::duration<double, std::milli>(warmup_end - warmup_start).count();

    m_warmed_up = true;
    GENAI_INFO("AutoencoderKLWan: Warmup completed in " + std::to_string(static_cast<int>(warmup_time_ms)) +
               " ms (JIT compilation done)");
}

void AutoencoderKLWan::enable_tiling(int tile_sample_min_height,
                                      int tile_sample_min_width,
                                      int tile_sample_stride_height,
                                      int tile_sample_stride_width) {
    m_tiling_config.enabled = true;
    m_tiling_config.tile_sample_min_height = tile_sample_min_height;
    m_tiling_config.tile_sample_min_width = tile_sample_min_width;
    m_tiling_config.tile_sample_stride_height = tile_sample_stride_height;
    m_tiling_config.tile_sample_stride_width = tile_sample_stride_width;

    GENAI_INFO("AutoencoderKLWan: Tiling enabled with tile_size=[" +
               std::to_string(tile_sample_min_height) + "x" + std::to_string(tile_sample_min_width) +
               "], stride=[" + std::to_string(tile_sample_stride_height) + "x" +
               std::to_string(tile_sample_stride_width) + "]");
}

void AutoencoderKLWan::disable_tiling() {
    m_tiling_config.enabled = false;
}

ov::Tensor AutoencoderKLWan::decode(ov::Tensor latents) {
    // latents shape: [B, C, T, H, W] for video
    auto shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 5, "AutoencoderKLWan expects 5D input tensor [B, C, T, H, W]");

    size_t batch = shape[0], channels = shape[1], num_frames = shape[2];
    size_t height = shape[3];
    size_t width = shape[4];

    // Calculate latent tile sizes
    int tile_latent_min_h = m_tiling_config.tile_sample_min_height / m_tiling_config.spatial_compression_ratio;
    int tile_latent_min_w = m_tiling_config.tile_sample_min_width / m_tiling_config.spatial_compression_ratio;

    // Check if tiling is needed
    if (m_tiling_config.enabled &&
        (height > static_cast<size_t>(tile_latent_min_h) || width > static_cast<size_t>(tile_latent_min_w))) {
        GENAI_INFO("AutoencoderKLWan: Using tiled decode for latent size [" +
                   std::to_string(height) + "x" + std::to_string(width) + "]");
        return tiled_decode(latents);
    }

    // Non-tiled decode - print statistics for comparison
    size_t out_h = height * 8, out_w = width * 8, out_t = num_frames * 4;
    size_t full_latent_bytes = batch * channels * num_frames * height * width * sizeof(float);
    size_t full_output_bytes = batch * 3 * out_t * out_h * out_w * sizeof(float);

    GENAI_INFO("========== VAE Non-Tiled Decode Statistics ==========");
    GENAI_INFO("Input latent shape: [" + std::to_string(batch) + ", " + std::to_string(channels) +
               ", " + std::to_string(num_frames) + ", " + std::to_string(height) + ", " + std::to_string(width) + "]");
    GENAI_INFO("Output video shape: [" + std::to_string(batch) + ", " + std::to_string(out_t) +
               ", " + std::to_string(out_h) + ", " + std::to_string(out_w) + ", 3]");
    GENAI_INFO("Tiling: DISABLED (full resolution decode)");
    GENAI_INFO("Memory usage:");
    GENAI_INFO("  - Latent input:  " + std::to_string(full_latent_bytes / 1024 / 1024) + " MB");
    GENAI_INFO("  - Output buffer: " + std::to_string(full_output_bytes / 1024 / 1024) + " MB");
    GENAI_INFO("======================================================");

    auto start_time = std::chrono::high_resolution_clock::now();
    ov::Tensor result = decode_single(latents);
    auto end_time = std::chrono::high_resolution_clock::now();
    double decode_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    GENAI_INFO("========== VAE Non-Tiled Decode Timing ==========");
    GENAI_INFO("  - Decode time: " + std::to_string(static_cast<int>(decode_time_ms)) + " ms");
    GENAI_INFO("Output shape: [" +
               std::to_string(result.get_shape()[0]) + ", " +
               std::to_string(result.get_shape()[1]) + ", " +
               std::to_string(result.get_shape()[2]) + ", " +
               std::to_string(result.get_shape()[3]) + ", " +
               std::to_string(result.get_shape()[4]) + "]");
    GENAI_INFO("==================================================");

    return result;
}

ov::Tensor AutoencoderKLWan::decode_single(ov::Tensor latents) {
    m_decoder_request.set_input_tensor(latents);
    {
        PROFILE(pm, "vae_decoder infer");
        m_decoder_request.infer();
    }
    ov::Tensor output = m_decoder_request.get_output_tensor();
    // Make a copy to avoid issues with tensor reuse
    ov::Tensor result(output.get_element_type(), output.get_shape());
    output.copy_to(result);
    return result;
}

ov::Tensor AutoencoderKLWan::slice_5d(const ov::Tensor& tensor,
                                       size_t h_start, size_t h_end,
                                       size_t w_start, size_t w_end) {
    // tensor shape: [B, C, T, H, W]
    auto shape = tensor.get_shape();
    size_t B = shape[0], C = shape[1], T = shape[2], H = shape[3], W = shape[4];

    size_t tile_h = h_end - h_start;
    size_t tile_w = w_end - w_start;

    ov::Tensor tile(tensor.get_element_type(), {B, C, T, tile_h, tile_w});

    const float* src = tensor.data<float>();
    float* dst = tile.data<float>();

    // Copy row by row for each batch, channel, and time frame
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

ov::Tensor AutoencoderKLWan::pad_tile_5d(const ov::Tensor& tile, size_t target_h, size_t target_w) {
    // Pad a latent tile [B, C, T, H, W] to fixed target size using reflection padding
    auto shape = tile.get_shape();
    size_t B = shape[0], C = shape[1], T = shape[2], H = shape[3], W = shape[4];

    if (H >= target_h && W >= target_w) {
        return tile;  // No padding needed
    }

    ov::Tensor padded(tile.get_element_type(), {B, C, T, target_h, target_w});

    const float* src = tile.data<float>();
    float* dst = padded.data<float>();

    // Initialize with zeros
    std::memset(dst, 0, padded.get_byte_size());

    // Copy original data and use reflection padding for edges
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t h = 0; h < target_h; ++h) {
                    for (size_t w = 0; w < target_w; ++w) {
                        // Reflection padding: mirror at boundaries
                        size_t src_h = h < H ? h : (2 * H - h - 2);
                        size_t src_w = w < W ? w : (2 * W - w - 2);

                        // Clamp to valid range
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

ov::Tensor AutoencoderKLWan::crop_decoded_tile_5d(const ov::Tensor& decoded,
                                                  size_t orig_latent_h, size_t orig_latent_w,
                                                  size_t full_latent_h, size_t full_latent_w) {
    // Crop decoded tile output to remove padding
    // decoded shape: [B, T, H, W, C] (u8) after postprocess
    // orig_latent_h/w: original latent tile size before padding
    // full_latent_h/w: padded latent tile size

    auto shape = decoded.get_shape();
    size_t B = shape[0], T = shape[1], H = shape[2], W = shape[3], C = shape[4];

    // Calculate expected output size from original latent (8x spatial upscale)
    size_t target_h = orig_latent_h * 8;
    size_t target_w = orig_latent_w * 8;

    if (H == target_h && W == target_w) {
        return decoded;  // No cropping needed
    }

    ov::Tensor cropped(decoded.get_element_type(), {B, T, target_h, target_w, C});

    const uint8_t* src = decoded.data<uint8_t>();
    uint8_t* dst = cropped.data<uint8_t>();

    // Copy only the valid region
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

ov::Tensor AutoencoderKLWan::concat_tiles_5d(const std::vector<std::vector<ov::Tensor>>& tiles,
                                              size_t tile_stride_h, size_t tile_stride_w) {
    if (tiles.empty() || tiles[0].empty()) {
        OPENVINO_THROW("AutoencoderKLWan: Empty tiles for concatenation");
    }

    // Get dimensions from first tile
    auto first_shape = tiles[0][0].get_shape();
    bool is_postprocessed = (tiles[0][0].get_element_type() == ov::element::u8);

    size_t num_rows = tiles.size();
    size_t num_cols = tiles[0].size();

    // Calculate output dimensions
    size_t B, T, C;
    size_t total_h, total_w;

    if (is_postprocessed) {
        // [B, T, H, W, C]
        B = first_shape[0];
        T = first_shape[1];
        C = first_shape[4];
        total_h = (num_rows - 1) * tile_stride_h + tiles.back()[0].get_shape()[2];
        total_w = (num_cols - 1) * tile_stride_w + tiles[0].back().get_shape()[3];
    } else {
        // [B, C, T, H, W]
        B = first_shape[0];
        C = first_shape[1];
        T = first_shape[2];
        total_h = (num_rows - 1) * tile_stride_h + tiles.back()[0].get_shape()[3];
        total_w = (num_cols - 1) * tile_stride_w + tiles[0].back().get_shape()[4];
    }

    ov::Shape result_shape;
    if (is_postprocessed) {
        result_shape = {B, T, total_h, total_w, C};
    } else {
        result_shape = {B, C, T, total_h, total_w};
    }

    ov::Tensor result(tiles[0][0].get_element_type(), result_shape);

    // Copy tiles to result
    size_t h_offset = 0;
    for (size_t row = 0; row < num_rows; ++row) {
        size_t w_offset = 0;
        size_t tile_h = is_postprocessed ? tiles[row][0].get_shape()[2] : tiles[row][0].get_shape()[3];
        size_t copy_h = (row == num_rows - 1) ? tile_h : tile_stride_h;

        for (size_t col = 0; col < num_cols; ++col) {
            const auto& tile = tiles[row][col];
            auto tile_shape = tile.get_shape();
            size_t tile_w = is_postprocessed ? tile_shape[3] : tile_shape[4];
            size_t copy_w = (col == num_cols - 1) ? tile_w : tile_stride_w;

            // Copy the valid portion of the tile
            if (is_postprocessed) {
                // [B, T, H, W, C]
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
                // [B, C, T, H, W]
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

            w_offset += tile_stride_w;
        }
        h_offset += tile_stride_h;
    }

    return result;
}

ov::Tensor AutoencoderKLWan::tiled_decode(ov::Tensor latents) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // latents shape: [B, C, T, H, W]
    auto shape = latents.get_shape();
    size_t batch = shape[0], channels = shape[1], num_frames = shape[2];
    size_t height = shape[3];
    size_t width = shape[4];

    // Calculate tile sizes in latent space
    int tile_latent_min_h = m_tiling_config.tile_sample_min_height / m_tiling_config.spatial_compression_ratio;
    int tile_latent_min_w = m_tiling_config.tile_sample_min_width / m_tiling_config.spatial_compression_ratio;
    int tile_latent_stride_h = m_tiling_config.tile_sample_stride_height / m_tiling_config.spatial_compression_ratio;
    int tile_latent_stride_w = m_tiling_config.tile_sample_stride_width / m_tiling_config.spatial_compression_ratio;

    // Calculate blend extent in sample space
    size_t blend_h = m_tiling_config.tile_sample_min_height - m_tiling_config.tile_sample_stride_height;
    size_t blend_w = m_tiling_config.tile_sample_min_width - m_tiling_config.tile_sample_stride_width;

    // Calculate number of tiles
    size_t num_rows = (height + tile_latent_stride_h - 1) / tile_latent_stride_h;
    size_t num_cols = (width + tile_latent_stride_w - 1) / tile_latent_stride_w;
    size_t total_tiles = num_rows * num_cols;

    // Estimate memory usage
    // GPU memory during VAE inference is the key metric
    // Full resolution: B * C * T * H * W * 4 bytes (f32) for latent input
    // Tiled: B * C * T * tile_H * tile_W * 4 bytes per tile
    size_t full_latent_bytes = batch * channels * num_frames * height * width * sizeof(float);
    size_t tile_latent_bytes = batch * channels * num_frames * tile_latent_min_h * tile_latent_min_w * sizeof(float);

    // Output size (8x upscale in H,W, approximately 2x in T for Wan 2.1)
    // Note: actual T factor depends on VAE architecture
    size_t out_h = height * 8, out_w = width * 8;
    // Estimate intermediate activation memory (VAE decoder has multiple conv layers)
    // This is a rough estimate: ~10x the output size for intermediate activations
    size_t full_intermediate_bytes = batch * 512 * num_frames * out_h * out_w * sizeof(float) / 4;  // estimate
    size_t tile_intermediate_bytes = batch * 512 * num_frames * m_tiling_config.tile_sample_min_height * m_tiling_config.tile_sample_min_width * sizeof(float) / 4;

    // Calculate spatial reduction ratio
    float latent_reduction = static_cast<float>(tile_latent_min_h * tile_latent_min_w) / (height * width);
    float gpu_memory_reduction_pct = (1.0f - latent_reduction) * 100.0f;

    // Helper for KB display when MB is 0
    auto format_bytes = [](size_t bytes) -> std::string {
        if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / 1024 / 1024) + " MB";
        } else {
            return std::to_string(bytes / 1024) + " KB";
        }
    };

    GENAI_INFO("========== VAE Tiled Decode Statistics ==========");
    GENAI_INFO("Input latent shape: [" + std::to_string(batch) + ", " + std::to_string(channels) +
               ", " + std::to_string(num_frames) + ", " + std::to_string(height) + ", " + std::to_string(width) + "]");
    GENAI_INFO("Output video shape: [" + std::to_string(batch) + ", T, " +
               std::to_string(out_h) + ", " + std::to_string(out_w) + ", 3]");
    GENAI_INFO("Tile configuration:");
    GENAI_INFO("  - Latent tile: " + std::to_string(tile_latent_min_h) + "x" + std::to_string(tile_latent_min_w) +
               " (vs full " + std::to_string(height) + "x" + std::to_string(width) + ")");
    GENAI_INFO("  - Sample tile: " + std::to_string(m_tiling_config.tile_sample_min_height) + "x" +
               std::to_string(m_tiling_config.tile_sample_min_width) +
               " (vs full " + std::to_string(out_h) + "x" + std::to_string(out_w) + ")");
    GENAI_INFO("  - Stride: " + std::to_string(tile_latent_stride_h) + "x" + std::to_string(tile_latent_stride_w) +
               " (latent), " + std::to_string(m_tiling_config.tile_sample_stride_height) + "x" +
               std::to_string(m_tiling_config.tile_sample_stride_width) + " (sample)");
    GENAI_INFO("  - Overlap: " + std::to_string(blend_h) + "x" + std::to_string(blend_w) + " pixels");
    GENAI_INFO("Tile grid: " + std::to_string(num_rows) + " rows x " + std::to_string(num_cols) +
               " cols = " + std::to_string(total_tiles) + " total tiles");
    GENAI_INFO("GPU memory estimate (per inference):");
    GENAI_INFO("  - Latent input: " + format_bytes(full_latent_bytes) + " (full) vs " +
               format_bytes(tile_latent_bytes) + " (tile)");
    GENAI_INFO("  - Intermediate activations: ~" + format_bytes(full_intermediate_bytes) + " (full) vs ~" +
               format_bytes(tile_intermediate_bytes) + " (tile)");
    GENAI_INFO("  - Spatial reduction: " + std::to_string(height * width) + " -> " +
               std::to_string(tile_latent_min_h * tile_latent_min_w) + " = " +
               std::to_string(static_cast<int>(gpu_memory_reduction_pct)) + "% less GPU memory per inference");
    GENAI_INFO("  - Using FIXED tile size with padding to avoid dynamic shape recompilation");
    GENAI_INFO("==================================================");

    // Split latent into overlapping tiles and decode separately
    // Use FIXED tile size with padding to avoid OpenVINO dynamic shape recompilation
    std::vector<std::vector<ov::Tensor>> rows;
    std::vector<std::vector<std::pair<size_t, size_t>>> orig_tile_sizes;  // Store original sizes for cropping
    size_t tile_idx = 0;
    double total_decode_time_ms = 0.0;

    for (size_t i = 0; i < height; i += tile_latent_stride_h) {
        std::vector<ov::Tensor> row;
        std::vector<std::pair<size_t, size_t>> row_sizes;

        for (size_t j = 0; j < width; j += tile_latent_stride_w) {
            size_t h_end = std::min(i + static_cast<size_t>(tile_latent_min_h), height);
            size_t w_end = std::min(j + static_cast<size_t>(tile_latent_min_w), width);

            size_t orig_tile_h = h_end - i;
            size_t orig_tile_w = w_end - j;
            bool needs_padding = (orig_tile_h < static_cast<size_t>(tile_latent_min_h) ||
                                  orig_tile_w < static_cast<size_t>(tile_latent_min_w));

            // Extract tile from latent
            ov::Tensor tile = slice_5d(latents, i, h_end, j, w_end);

            // Pad to fixed size if needed to avoid dynamic shape recompilation
            if (needs_padding) {
                tile = pad_tile_5d(tile, tile_latent_min_h, tile_latent_min_w);
            }

            auto tile_start = std::chrono::high_resolution_clock::now();

            // Decode this tile (always fixed size now)
            ov::Tensor decoded_tile = decode_single(tile);

            auto tile_end = std::chrono::high_resolution_clock::now();
            double tile_time_ms = std::chrono::duration<double, std::milli>(tile_end - tile_start).count();
            total_decode_time_ms += tile_time_ms;

            // Crop decoded tile to remove padding if it was added
            if (needs_padding) {
                decoded_tile = crop_decoded_tile_5d(decoded_tile, orig_tile_h, orig_tile_w,
                                                    tile_latent_min_h, tile_latent_min_w);
            }

            tile_idx++;
            std::string pad_info = needs_padding ? " (padded)" : "";
            GENAI_INFO("  Tile " + std::to_string(tile_idx) + "/" + std::to_string(total_tiles) +
                       " [" + std::to_string(i) + ":" + std::to_string(h_end) +
                       ", " + std::to_string(j) + ":" + std::to_string(w_end) + "]" + pad_info + " - " +
                       std::to_string(static_cast<int>(tile_time_ms)) + " ms");

            row.push_back(decoded_tile);
            row_sizes.push_back({orig_tile_h, orig_tile_w});
        }
        rows.push_back(row);
        orig_tile_sizes.push_back(row_sizes);
    }

    // Blend overlapping tiles (CPU)
    auto blend_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < rows[i].size(); ++j) {
            // Blend with tile above
            if (i > 0) {
                blend_utils::blend_v_5d(rows[i - 1][j], rows[i][j], blend_h);
            }
            // Blend with tile to the left
            if (j > 0) {
                blend_utils::blend_h_5d(rows[i][j - 1], rows[i][j], blend_w);
            }
        }
    }

    auto blend_end = std::chrono::high_resolution_clock::now();
    double blend_time_ms = std::chrono::duration<double, std::milli>(blend_end - blend_start).count();

    // Calculate output stride in sample space
    size_t sample_stride_h = m_tiling_config.tile_sample_stride_height;
    size_t sample_stride_w = m_tiling_config.tile_sample_stride_width;

    // Concatenate all tiles
    auto concat_start = std::chrono::high_resolution_clock::now();
    ov::Tensor result = concat_tiles_5d(rows, sample_stride_h, sample_stride_w);
    auto concat_end = std::chrono::high_resolution_clock::now();
    double concat_time_ms = std::chrono::duration<double, std::milli>(concat_end - concat_start).count();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    GENAI_INFO("========== VAE Tiled Decode Timing ==========");
    GENAI_INFO("  - Total tiles decoded: " + std::to_string(total_tiles));
    GENAI_INFO("  - Decode time: " + std::to_string(static_cast<int>(total_decode_time_ms)) + " ms (" +
               std::to_string(static_cast<int>(total_decode_time_ms / total_tiles)) + " ms/tile avg)");
    GENAI_INFO("  - Blend time:  " + std::to_string(static_cast<int>(blend_time_ms)) + " ms");
    GENAI_INFO("  - Concat time: " + std::to_string(static_cast<int>(concat_time_ms)) + " ms");
    GENAI_INFO("  - Total time:  " + std::to_string(static_cast<int>(total_time_ms)) + " ms");
    GENAI_INFO("Output shape: [" +
               std::to_string(result.get_shape()[0]) + ", " +
               std::to_string(result.get_shape()[1]) + ", " +
               std::to_string(result.get_shape()[2]) + ", " +
               std::to_string(result.get_shape()[3]) + ", " +
               std::to_string(result.get_shape()[4]) + "]");
    GENAI_INFO("=============================================");

    return result;
}

}
