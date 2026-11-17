// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <vector>

#include "json_utils.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include "video_generation/video2video_pipeline_impl.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

/// FlashVSR (one-step streaming video super-resolution, x4) in the layout of `genai-cli export`: lq_projector,
/// transformer (stateful key / value cache) and vae_decoder (stateful memories).
///
/// Streaming plan (upstream FlashVSRTinyPipeline): latent frame j is conditioned on the input frames [4j + 1, 4j + 5)
/// (LQ projector), decodes to the output frames [4j - 3, 4j + 1) with the input frames of the same range as decoder
/// condition (frames before 0 are frame 0), and the transformer runs on chunks of first_chunk_frames latent frames,
/// then chunk_frames ones. The input frames are the low-resolution frames upscaled with Pillow's bicubic filter and
/// center cropped to a multiple of the attention block size.
class FlashVSRPipeline : public VideoToVideoPipelineImpl {
public:
    explicit FlashVSRPipeline(const std::filesystem::path& root) {
        const auto read_config = [&](const std::filesystem::path& path) {
            std::ifstream file(path);
            OPENVINO_ASSERT(file.is_open(), "Failed to open ", path);
            return nlohmann::json::parse(file);
        };
        using utils::read_json_param;
        const auto index = read_config(root / "model_index.json");
        read_json_param(index, "scale_factor", m_scale);

        const auto transformer = read_config(root / "transformer" / "config.json");
        std::vector<size_t> block_size;
        read_json_param(transformer, "block_size", block_size);
        OPENVINO_ASSERT(block_size.size() == 3, "transformer/config.json: block_size must have 3 values");
        read_json_param(transformer, "first_chunk_frames", m_first_chunk_latents);
        read_json_param(transformer, "chunk_frames", m_chunk_latents);
        read_json_param(transformer, "sparse_reference_area", m_sparse_reference_area);
        read_json_param(transformer, "in_channels", m_latent_channels);

        const auto lq = read_config(root / "lq_projector" / "config.json");
        read_json_param(lq, "frames_per_latent", m_frames_per_latent);
        read_json_param(lq, "patch_size", m_lq_patch);

        const auto decoder = read_config(root / "vae_decoder" / "config.json");
        read_json_param(decoder, "spatial_upscale", m_vae_scale);
        read_json_param(decoder, "temporal_upscale", m_temporal_upscale);
        OPENVINO_ASSERT(m_temporal_upscale == m_frames_per_latent,
                        "FlashVSR: the decoder must give one output frame per input frame of the LQ projector");

        m_block_h = block_size[1] * m_lq_patch;
        m_block_w = block_size[2] * m_lq_patch;

        auto& core = utils::singleton_core();
        m_lq_model = core.read_model(root / "lq_projector" / "openvino_model.xml");
        m_transformer_model = core.read_model(root / "transformer" / "openvino_model.xml");
        m_decoder_model = core.read_model(root / "vae_decoder" / "openvino_model.xml");

        m_generation_config = VideoGenerationConfig{};
        replace_defaults(m_generation_config);
    }

    FlashVSRPipeline(const std::filesystem::path& root, const std::string& device, const ov::AnyMap& properties)
        : FlashVSRPipeline(root) {
        compile(device, properties);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        auto& core = utils::singleton_core();
        m_lq = core.compile_model(m_lq_model, device, properties).create_infer_request();
        m_transformer = core.compile_model(m_transformer_model, device, properties).create_infer_request();
        m_decoder = core.compile_model(m_decoder_model, device, properties).create_infer_request();
        m_lq_model.reset();
        m_transformer_model.reset();
        m_decoder_model.reset();
        m_compiled = true;
    }

    void start_stream(const ov::AnyMap& properties, size_t total_frames) override {
        OPENVINO_ASSERT(m_compiled, "VideoToVideoPipeline is not compiled: call compile() first");
        m_config = merge_generation_config(properties);
        OPENVINO_ASSERT(m_config.generator, "VideoToVideoPipeline: the random generator must not be null");
        m_frames.clear();
        m_frames_base = 0;
        m_pushed = 0;
        m_next_latent = 0;
        m_chunk_index = 0;
        m_emitted = 0;
        m_total_frames = total_frames;
        m_in_h = m_in_w = 0;
        m_noise = ov::Tensor();
        m_transformer.reset_state();
        m_decoder.reset_state();
        m_perf_metrics.clean_up();
        m_stream_start = std::chrono::steady_clock::now();
        m_streaming = true;
    }

    ov::Tensor push_frames(const ov::Tensor& frames) override {
        OPENVINO_ASSERT(m_streaming, "VideoToVideoPipeline: call start_stream() before push_frames()");
        OPENVINO_ASSERT(frames.get_element_type() == ov::element::u8, "VideoToVideoPipeline expects uint8 frames");
        const auto& shape = frames.get_shape();
        OPENVINO_ASSERT((shape.size() == 4 || shape.size() == 3) && shape.back() == 3,
                        "VideoToVideoPipeline expects RGB frames [N, H, W, 3] or [H, W, 3], got ",
                        shape);
        const size_t count = shape.size() == 4 ? shape[0] : 1;
        const size_t h = shape[shape.size() - 3];
        const size_t w = shape[shape.size() - 2];
        if (m_in_h == 0) {
            set_input_size(h, w);
        }
        OPENVINO_ASSERT(h == m_in_h && w == m_in_w,
                        "VideoToVideoPipeline: all frames of a stream must have the same size");
        std::vector<ov::Tensor> outputs;
        const auto* data = frames.data<const uint8_t>();
        for (size_t i = 0; i < count; ++i) {
            add_frame(data + i * h * w * 3);
            run_ready_chunks(outputs);
        }
        return concat_frames(outputs);
    }

    ov::Tensor end_stream() override {
        OPENVINO_ASSERT(m_streaming, "VideoToVideoPipeline: call start_stream() before end_stream()");
        std::vector<ov::Tensor> outputs;
        const size_t real = m_pushed;
        if (real > 0) {
            // Repeat the last frame until every real frame is decoded.
            while (covered_frames() < real) {
                const auto last = m_frames.back();
                m_frames.push_back(last);
                ++m_pushed;
                run_ready_chunks(outputs);
            }
        }
        m_streaming = false;
        ov::Tensor result = concat_frames(outputs);
        // Drop the outputs of the repeated frames.
        const size_t emitted_before = m_emitted - (result ? result.get_shape()[0] : 0);
        const size_t keep =
            real > emitted_before ? std::min(real - emitted_before, result ? result.get_shape()[0] : 0) : 0;
        m_perf_metrics.generate_duration = Ms{std::chrono::steady_clock::now() - m_stream_start}.count();
        if (!result || keep == result.get_shape()[0]) {
            return result ? result : empty_frames();
        }
        ov::Shape kept_shape = result.get_shape();
        kept_shape[0] = keep;
        ov::Tensor kept(ov::element::u8, kept_shape);
        std::memcpy(kept.data(), result.data(), kept.get_byte_size());
        return kept;
    }

protected:
    void replace_defaults(VideoGenerationConfig& config) const override {
        if (!config.sparse_attention_ratio) {
            config.sparse_attention_ratio = 2.0f;
        }
        if (!config.color_correction) {
            config.color_correction = true;
        }
        if (!config.generator) {
            config.generator = std::make_shared<CppStdGenerator>(42);
        }
    }

private:
    void set_input_size(size_t h, size_t w) {
        m_in_h = h;
        m_in_w = w;
        m_scaled_h = h * m_scale;
        m_scaled_w = w * m_scale;
        m_out_h = m_scaled_h / m_block_h * m_block_h;
        m_out_w = m_scaled_w / m_block_w * m_block_w;
        OPENVINO_ASSERT(m_out_h > 0 && m_out_w > 0,
                        "VideoToVideoPipeline: the upscaled frames (",
                        m_scaled_w,
                        "x",
                        m_scaled_h,
                        ") must be at least ",
                        m_block_w,
                        "x",
                        m_block_h);
        m_crop_top = (m_scaled_h - m_out_h) / 2;
        m_crop_left = (m_scaled_w - m_out_w) / 2;
        // Upstream topk = int(G * G * topk_ratio) - 1, topk_ratio = sparse_ratio * reference_area / (H * W).
        const double blocks = static_cast<double>(m_out_h / m_block_h) * static_cast<double>(m_out_w / m_block_w);
        const double topk_ratio = static_cast<double>(*m_config.sparse_attention_ratio) *
                                  static_cast<double>(m_sparse_reference_area) /
                                  (static_cast<double>(m_out_h) * static_cast<double>(m_out_w));
        m_top_k = static_cast<int64_t>(blocks * blocks * topk_ratio) - 1;
        if (m_total_frames > 0) {
            // The whole noise at once, as the reference implementation draws it.
            m_noise = m_config.generator->randn_tensor(ov::Shape{1,
                                                                 m_latent_channels,
                                                                 latents_for_frames(m_total_frames),
                                                                 m_out_h / m_vae_scale,
                                                                 m_out_w / m_vae_scale});
        }
    }

    /// Number of latent frames of the chunks needed to decode `frames` output frames.
    size_t latents_for_frames(size_t frames) const {
        size_t latents = m_first_chunk_latents;
        while (latents * m_frames_per_latent - (m_frames_per_latent - 1) < frames) {
            latents += m_chunk_latents;
        }
        return latents;
    }

    /// Number of output frames decoded so far (including the outputs of repeated frames).
    size_t covered_frames() const {
        return m_next_latent == 0 ? 0 : m_next_latent * m_frames_per_latent - (m_frames_per_latent - 1);
    }

    void add_frame(const uint8_t* rgb) {
        clip_image_u8 src{static_cast<int>(m_in_w),
                          static_cast<int>(m_in_h),
                          std::vector<uint8_t>(rgb, rgb + m_in_h * m_in_w * 3)};
        clip_image_u8 up;
        bicubic_resize(src, up, static_cast<int>(m_scaled_w), static_cast<int>(m_scaled_h));
        // Planar [3, H, W] float in [-1, 1] of the center crop.
        std::vector<float> frame(3 * m_out_h * m_out_w);
        const size_t plane = m_out_h * m_out_w;
        for (size_t y = 0; y < m_out_h; ++y) {
            const uint8_t* row = up.buf.data() + ((y + m_crop_top) * m_scaled_w + m_crop_left) * 3;
            for (size_t x = 0; x < m_out_w; ++x) {
                for (size_t c = 0; c < 3; ++c) {
                    frame[c * plane + y * m_out_w + x] = static_cast<float>(row[x * 3 + c]) / 255.0f * 2.0f - 1.0f;
                }
            }
        }
        m_frames.push_back(std::move(frame));
        ++m_pushed;
    }

    const std::vector<float>& frame_at(int64_t index) const {
        const auto clamped = static_cast<size_t>(std::max<int64_t>(index, 0));
        OPENVINO_ASSERT(clamped >= m_frames_base && clamped - m_frames_base < m_frames.size(),
                        "FlashVSR: frame ",
                        index,
                        " is not buffered");
        return m_frames[clamped - m_frames_base];
    }

    /// [1, 3, end - begin, H, W] f32 tensor of the input frames [begin, end).
    ov::Tensor frames_tensor(int64_t begin, int64_t end) const {
        const size_t count = static_cast<size_t>(end - begin);
        const size_t plane = m_out_h * m_out_w;
        ov::Tensor t(ov::element::f32, ov::Shape{1, 3, count, m_out_h, m_out_w});
        auto* dst = t.data<float>();
        for (size_t i = 0; i < count; ++i) {
            const auto& frame = frame_at(begin + static_cast<int64_t>(i));
            for (size_t c = 0; c < 3; ++c) {
                std::memcpy(dst + (c * count + i) * plane, frame.data() + c * plane, plane * sizeof(float));
            }
        }
        return t;
    }

    void run_ready_chunks(std::vector<ov::Tensor>& outputs) {
        for (;;) {
            const size_t latents = m_chunk_index == 0 ? m_first_chunk_latents : m_chunk_latents;
            const size_t j0 = m_next_latent;
            const size_t j1 = j0 + latents;
            // The LQ projector needs the input frames up to 4 * j1 + 1 (exclusive).
            if (m_pushed < j1 * m_frames_per_latent + 1) {
                return;
            }
            outputs.push_back(run_chunk(j0, j1));
            m_next_latent = j1;
            ++m_chunk_index;
            // The next chunk reads the input frames from 4 * j1 - 3 on.
            const int64_t keep_from =
                static_cast<int64_t>(j1 * m_frames_per_latent) - static_cast<int64_t>(m_frames_per_latent - 1);
            while (static_cast<int64_t>(m_frames_base) < keep_from && m_frames.size() > 1) {
                m_frames.pop_front();
                ++m_frames_base;
            }
        }
    }

    ov::Tensor run_chunk(size_t j0, size_t j1) {
        const auto start = std::chrono::steady_clock::now();
        const size_t f = j1 - j0;
        const auto fpl = static_cast<int64_t>(m_frames_per_latent);
        const auto lat0 = static_cast<int64_t>(j0);
        const auto lat1 = static_cast<int64_t>(j1);

        m_lq.set_tensor("lq_frames", frames_tensor(fpl * lat0 + 1, fpl * lat1 + 1));
        m_lq.infer();
        const ov::Tensor lq_latents = m_lq.get_tensor("lq_latents");

        const size_t lh = m_out_h / m_vae_scale, lw = m_out_w / m_vae_scale;
        ov::Tensor noise(ov::element::f32, ov::Shape{1, m_latent_channels, f, lh, lw});
        if (m_noise) {
            // Slice [j0, j1) of the whole noise [1, C, T, h, w].
            const size_t total = m_noise.get_shape()[2];
            OPENVINO_ASSERT(j1 <= total,
                            "FlashVSR: the stream is longer than the number of frames given to generate()");
            const size_t plane = lh * lw;
            for (size_t c = 0; c < m_latent_channels; ++c) {
                std::memcpy(noise.data<float>() + c * f * plane,
                            m_noise.data<const float>() + (c * total + j0) * plane,
                            f * plane * sizeof(float));
            }
        } else {
            m_config.generator->randn_tensor(noise.get_shape()).copy_to(noise);
        }

        ov::Tensor frame_offset(ov::element::i64, ov::Shape{});
        frame_offset.data<int64_t>()[0] = lat0;
        ov::Tensor top_k(ov::element::i64, ov::Shape{});
        top_k.data<int64_t>()[0] = m_top_k;
        m_transformer.set_tensor("latents", noise);
        m_transformer.set_tensor("lq_latents", lq_latents);
        m_transformer.set_tensor("frame_offset", frame_offset);
        m_transformer.set_tensor("top_k", top_k);
        const auto dit_start = std::chrono::steady_clock::now();
        m_transformer.infer();
        m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(
            MicroSeconds(std::chrono::steady_clock::now() - dit_start));

        // One-step flow matching from sigma 1: x0 = noise - v.
        const ov::Tensor noise_pred = m_transformer.get_tensor("noise_pred");
        ov::Tensor latents(ov::element::f32, noise.get_shape());
        {
            const float* n = noise.data<const float>();
            const float* v = noise_pred.data<const float>();
            float* x = latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); ++i) {
                x[i] = n[i] - v[i];
            }
        }

        // Output frames [4 j0 - 3, 4 j1 - 3), the ones before frame 0 are dropped.
        const int64_t first = fpl * lat0 - (fpl - 1);
        const size_t skip = first < 0 ? static_cast<size_t>(-first) : 0;
        const size_t count = f * m_frames_per_latent - skip;
        ov::Tensor out(ov::element::u8, ov::Shape{count, m_out_h, m_out_w, 3});

        // The decoder runs latent frame by latent frame (its memories are states): the activations of a whole chunk at
        // the output resolution would take several GB of device memory.
        const size_t latent_plane = lh * lw;
        size_t written = 0;
        for (size_t t = 0; t < f; ++t) {
            ov::Tensor latent(ov::element::f32, ov::Shape{1, m_latent_channels, 1, lh, lw});
            for (size_t c = 0; c < m_latent_channels; ++c) {
                std::memcpy(latent.data<float>() + c * latent_plane,
                            latents.data<const float>() + (c * f + t) * latent_plane,
                            latent_plane * sizeof(float));
            }
            const int64_t frame0 = first + static_cast<int64_t>(t * m_frames_per_latent);
            m_decoder.set_tensor("latents", latent);
            m_decoder.set_tensor("lq_frames", frames_tensor(frame0, frame0 + fpl));
            const auto vae_start = std::chrono::steady_clock::now();
            m_decoder.infer();
            m_perf_metrics.vae_decoder_inference_duration += Ms{std::chrono::steady_clock::now() - vae_start}.count();
            const ov::Tensor decoded = m_decoder.get_tensor("frames");  // [1, 3, 4, H, W] in about [-1, 1]
            const size_t frame_skip = frame0 < 0 ? std::min(static_cast<size_t>(-frame0), m_frames_per_latent) : 0;
            const size_t frames_out = m_frames_per_latent - frame_skip;
            if (frames_out == 0) {
                continue;
            }
            ov::Tensor part(out,
                            ov::Coordinate{written, 0, 0, 0},
                            ov::Coordinate{written + frames_out, m_out_h, m_out_w, 3});
            postprocess(decoded, m_frames_per_latent, frame_skip, frame0 + static_cast<int64_t>(frame_skip), part);
            written += frames_out;
        }
        m_emitted += count;
        m_perf_metrics.raw_metrics.iteration_durations.emplace_back(
            MicroSeconds(std::chrono::steady_clock::now() - start));
        return out;
    }

    /// Color correction against the input frames and conversion to uint8 NHWC.
    void postprocess(const ov::Tensor& decoded,
                     size_t frames,
                     size_t skip,
                     int64_t first_frame,
                     ov::Tensor& out) const {
        const size_t plane = m_out_h * m_out_w;
        const float* src = decoded.data<const float>();
        uint8_t* dst = out.data<uint8_t>();
        std::vector<float> fixed(plane);
        for (size_t t = skip; t < frames; ++t) {
            const size_t o = t - skip;
            const auto& lq = frame_at(first_frame + static_cast<int64_t>(o));
            for (size_t c = 0; c < 3; ++c) {
                const float* hq = src + (c * frames + t) * plane;
                const float* ref = lq.data() + c * plane;
                if (*m_config.color_correction) {
                    // AdaIN: per channel mean / std (biased variance + 1e-5) of the generated frame matched to the
                    // input.
                    double hm = 0.0, hv = 0.0, rm = 0.0, rv = 0.0;
                    for (size_t i = 0; i < plane; ++i) {
                        hm += hq[i];
                        rm += ref[i];
                    }
                    hm /= static_cast<double>(plane);
                    rm /= static_cast<double>(plane);
                    for (size_t i = 0; i < plane; ++i) {
                        hv += (hq[i] - hm) * (hq[i] - hm);
                        rv += (ref[i] - rm) * (ref[i] - rm);
                    }
                    const double hs = std::sqrt(hv / static_cast<double>(plane) + 1e-5);
                    const double rs = std::sqrt(rv / static_cast<double>(plane) + 1e-5);
                    for (size_t i = 0; i < plane; ++i) {
                        fixed[i] = static_cast<float>(std::clamp((hq[i] - hm) / hs * rs + rm, -1.0, 1.0));
                    }
                } else {
                    std::copy(hq, hq + plane, fixed.begin());
                }
                uint8_t* frame = dst + o * plane * 3;
                for (size_t i = 0; i < plane; ++i) {
                    // ((x + 1) * 127.5) clipped to [0, 255] and truncated, as the reference.
                    frame[i * 3 + c] = static_cast<uint8_t>(std::clamp((fixed[i] + 1.0f) * 127.5f, 0.0f, 255.0f));
                }
            }
        }
    }

    ov::Tensor empty_frames() const {
        return ov::Tensor(ov::element::u8, ov::Shape{0, m_out_h, m_out_w, 3});
    }

    ov::Tensor concat_frames(const std::vector<ov::Tensor>& parts) const {
        size_t count = 0;
        for (const auto& p : parts) {
            count += p.get_shape()[0];
        }
        ov::Tensor out(ov::element::u8, ov::Shape{count, m_out_h, m_out_w, 3});
        auto* dst = out.data<uint8_t>();
        for (const auto& p : parts) {
            std::memcpy(dst, p.data(), p.get_byte_size());
            dst += p.get_byte_size();
        }
        return out;
    }

    // Model geometry
    size_t m_scale = 4;
    size_t m_first_chunk_latents = 6;
    size_t m_chunk_latents = 2;
    size_t m_frames_per_latent = 4;
    size_t m_lq_patch = 16;
    size_t m_vae_scale = 8;
    size_t m_temporal_upscale = 4;
    size_t m_latent_channels = 16;
    size_t m_block_h = 128;
    size_t m_block_w = 128;
    int64_t m_sparse_reference_area = 768 * 1280;

    std::shared_ptr<ov::Model> m_lq_model, m_transformer_model, m_decoder_model;
    ov::InferRequest m_lq, m_transformer, m_decoder;
    bool m_compiled = false;

    // Stream state
    VideoGenerationConfig m_config;
    bool m_streaming = false;
    std::deque<std::vector<float>> m_frames;  // input frames from m_frames_base on, planar [3, H, W] in [-1, 1]
    size_t m_frames_base = 0;
    size_t m_pushed = 0;
    size_t m_next_latent = 0;
    size_t m_chunk_index = 0;
    size_t m_emitted = 0;
    size_t m_total_frames = 0;
    size_t m_in_h = 0, m_in_w = 0, m_scaled_h = 0, m_scaled_w = 0, m_out_h = 0, m_out_w = 0, m_crop_top = 0,
           m_crop_left = 0;
    int64_t m_top_k = 0;
    ov::Tensor m_noise;
    std::chrono::steady_clock::time_point m_stream_start;
};

}  // namespace ov::genai
