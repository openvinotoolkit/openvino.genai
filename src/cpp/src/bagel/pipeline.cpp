// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/bagel/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>

#include "bagel/npu/npu_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {

using Clock = std::chrono::steady_clock;

float ms_since(Clock::time_point t0) {
    return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
}

// numpy.random.RandomState(seed).randn: MT19937 with the legacy polar Box-Muller transform
class NumpyRandomState {
public:
    explicit NumpyRandomState(uint32_t seed) : m_mt(seed) {}

    double gauss() {
        if (m_has_gauss) {
            m_has_gauss = false;
            return m_gauss;
        }
        double x1, x2, r2;
        do {
            x1 = 2.0 * next_double() - 1.0;
            x2 = 2.0 * next_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        m_gauss = f * x1;
        m_has_gauss = true;
        return f * x2;
    }

private:
    double next_double() {
        const uint32_t a = m_mt() >> 5;
        const uint32_t b = m_mt() >> 6;
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }

    std::mt19937 m_mt;
    bool m_has_gauss = false;
    double m_gauss = 0.0;
};

// Python round(): half to even
int py_round(double v) {
    return static_cast<int>(std::nearbyint(v));
}

// BAGEL ImageTransform: MaxLongEdgeMinShortEdgeResize (bicubic), then ToTensor and Normalize(0.5, 0.5)
struct ImageTransform {
    int max_size;
    int min_size;
    int stride;
    double max_pixels;

    int make_divisible(int v) const {
        return std::max(stride, py_round(static_cast<double>(v) / stride) * stride);
    }

    std::pair<int, int> apply_scale(int w, int h, double scale) const {
        return {make_divisible(py_round(w * scale)), make_divisible(py_round(h * scale))};
    }

    clip_image_u8 resize(const clip_image_u8& src) const {
        const int w = src.nx, h = src.ny;
        double scale = std::min(static_cast<double>(max_size) / std::max(w, h), 1.0);
        scale = std::max(scale, static_cast<double>(min_size) / std::min(w, h));
        auto [nw, nh] = apply_scale(w, h, scale);
        if (static_cast<double>(nw) * nh > max_pixels) {
            std::tie(nw, nh) = apply_scale(nw, nh, max_pixels / (static_cast<double>(nw) * nh));
        }
        if (std::max(nw, nh) > max_size) {
            std::tie(nw, nh) = apply_scale(nw, nh, static_cast<double>(max_size) / std::max(nw, nh));
        }
        clip_image_u8 dst;
        bicubic_resize(src, dst, nw, nh);
        return dst;
    }
};

clip_image_u8 to_clip_image(const ov::Tensor& image) {
    const auto& shape = image.get_shape();
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 &&
                        (shape.size() == 3 || (shape.size() == 4 && shape[0] == 1)) && shape.back() == 3,
                    "BAGEL expects a u8 RGB image with [1, H, W, 3] or [H, W, 3] layout");
    const int h = static_cast<int>(shape[shape.size() - 3]);
    const int w = static_cast<int>(shape[shape.size() - 2]);
    OPENVINO_ASSERT(h > 0 && w > 0, "BAGEL expects a non-empty image");
    return {w, h, {image.data<uint8_t>(), image.data<uint8_t>() + image.get_size()}};
}

// [3, H, W] f32 in [-1, 1]
ov::Tensor to_normalized_tensor(const clip_image_u8& img) {
    ov::Tensor out(ov::element::f32, {3, static_cast<size_t>(img.ny), static_cast<size_t>(img.nx)});
    float* o = out.data<float>();
    const size_t plane = static_cast<size_t>(img.ny) * img.nx;
    for (size_t i = 0; i < plane; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            o[c * plane + i] = (static_cast<float>(img.buf[i * 3 + c]) / 255.0f - 0.5f) / 0.5f;
        }
    }
    return out;
}

// [C, H, W] -> [(H / p) * (W / p), p * p * C] in (row, col, py, px, c) order; crops to the given grid
ov::Tensor patchify(const float* src, size_t c, size_t h, size_t w, size_t p, size_t grid_h, size_t grid_w) {
    ov::Tensor out(ov::element::f32, {grid_h * grid_w, p * p * c});
    float* o = out.data<float>();
    for (size_t gh = 0; gh < grid_h; ++gh) {
        for (size_t gw = 0; gw < grid_w; ++gw) {
            for (size_t py = 0; py < p; ++py) {
                for (size_t px = 0; px < p; ++px) {
                    for (size_t ch = 0; ch < c; ++ch) {
                        *o++ = src[(ch * h + gh * p + py) * w + gw * p + px];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<int64_t> flat_position_ids(size_t grid_h, size_t grid_w, size_t max_per_side) {
    std::vector<int64_t> ids;
    ids.reserve(grid_h * grid_w);
    for (size_t r = 0; r < grid_h; ++r) {
        for (size_t c = 0; c < grid_w; ++c) {
            ids.push_back(static_cast<int64_t>(r * max_per_side + c));
        }
    }
    return ids;
}

ov::Tensor i64_tensor(const std::vector<int64_t>& values) {
    ov::Tensor t(ov::element::i64, {values.size()});
    std::copy(values.begin(), values.end(), t.data<int64_t>());
    return t;
}

std::vector<int64_t> iota_i64(int64_t begin, int64_t end) {
    std::vector<int64_t> v(static_cast<size_t>(std::max<int64_t>(end - begin, 0)));
    std::iota(v.begin(), v.end(), begin);
    return v;
}

ov::Tensor copy_tensor(const ov::Tensor& src) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    if (src.get_size() > 0) {
        std::memcpy(dst.data(), src.data(), src.get_byte_size());
    }
    return dst;
}

ov::Tensor causal_mask(size_t q, size_t past) {
    ov::Tensor m(ov::element::f32, {1, 1, q, past + q});
    float* d = m.data<float>();
    std::fill_n(d, m.get_size(), 0.0f);
    for (size_t r = 0; r < q; ++r) {
        std::fill(d + r * (past + q) + past + r + 1, d + (r + 1) * (past + q), -std::numeric_limits<float>::infinity());
    }
    return m;
}

ov::Tensor full_mask(size_t q, size_t past) {
    ov::Tensor m(ov::element::f32, {1, 1, q, past + q});
    std::fill_n(m.data<float>(), m.get_size(), 0.0f);
    return m;
}

float l2_norm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) {
        s += static_cast<double>(x) * x;
    }
    return static_cast<float>(std::sqrt(s));
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open ", path.string());
    return nlohmann::json::parse(f);
}

// optimum-intel exports name the models openvino_<name>; the original BAGEL IR set uses <name> directly
std::string model_file(const std::filesystem::path& models_dir, const std::string& name) {
    return std::filesystem::exists(models_dir / ("openvino_" + name)) ? "openvino_" + name : name;
}

struct BagelConfig {
    size_t hidden, layers, kv_heads, head_dim, latent_channel, latent_patch, latent_downsample, vit_patch, vit_max_side,
        max_latent;
    int64_t bos, eos, soi, eoi;
    // BagelImageProcessor defaults; an optimum-intel export stores its own
    ImageTransform vit_transform{980, 224, 14, 14.0 * 14 * 9 * 1024};
    ImageTransform vae_transform{1024, 512, 16, 14.0 * 14 * 9 * 1024};
};

// preprocessor_config.json, or the image_processor entry of processor_config.json
void read_image_processor(const std::filesystem::path& models_dir, BagelConfig& c) {
    nlohmann::json cfg;
    if (std::filesystem::exists(models_dir / "preprocessor_config.json")) {
        cfg = read_json(models_dir / "preprocessor_config.json");
    } else if (std::filesystem::exists(models_dir / "processor_config.json")) {
        cfg = read_json(models_dir / "processor_config.json").value("image_processor", nlohmann::json::object());
    }
    if (cfg.empty()) {
        return;
    }
    const double max_pixels = cfg.value("max_pixels", c.vit_transform.max_pixels);
    const auto& size = cfg.value("size", nlohmann::json::object());
    c.vit_transform = {size.value("longest_edge", c.vit_transform.max_size),
                       size.value("shortest_edge", c.vit_transform.min_size),
                       cfg.value("patch_size", c.vit_transform.stride),
                       max_pixels};
    c.vae_transform = {cfg.value("vae_max_size", c.vae_transform.max_size),
                       cfg.value("vae_min_size", c.vae_transform.min_size),
                       cfg.value("vae_stride", c.vae_transform.stride),
                       max_pixels};
}

// bagel_ov_config.json of the original IR set, or the Hugging Face config.json of an optimum-intel export
BagelConfig read_bagel_config(const std::filesystem::path& models_dir) {
    BagelConfig c;
    if (std::filesystem::exists(models_dir / "bagel_ov_config.json")) {
        const auto cfg = read_json(models_dir / "bagel_ov_config.json");
        const auto& ids = cfg.at("new_token_ids");
        c = {cfg.at("hidden_size").get<size_t>(),
             cfg.at("num_layers").get<size_t>(),
             cfg.at("num_kv_heads").get<size_t>(),
             cfg.at("head_dim").get<size_t>(),
             cfg.at("latent_channel").get<size_t>(),
             cfg.at("latent_patch_size").get<size_t>(),
             cfg.at("latent_downsample").get<size_t>(),
             cfg.at("vit_patch_size").get<size_t>(),
             cfg.at("vit_max_num_patch_per_side").get<size_t>(),
             cfg.at("max_latent_size").get<size_t>(),
             ids.at("bos_token_id").get<int64_t>(),
             ids.at("eos_token_id").get<int64_t>(),
             ids.at("start_of_image").get<int64_t>(),
             ids.at("end_of_image").get<int64_t>()};
        return c;
    }
    const auto cfg = read_json(models_dir / "config.json");
    OPENVINO_ASSERT(cfg.value("model_type", "") == "bagel",
                    "BagelPipeline expects a BAGEL model, got model_type '",
                    cfg.value("model_type", ""),
                    "' in ",
                    (models_dir / "config.json").string());
    const auto& text = cfg.at("text_config");
    const auto& vision = cfg.at("vision_config");
    const auto& vae = cfg.at("vae_config");
    const auto hidden = text.at("hidden_size").get<size_t>();
    const auto head_dim = text.contains("head_dim") && !text.at("head_dim").is_null()
                              ? text.at("head_dim").get<size_t>()
                              : hidden / text.at("num_attention_heads").get<size_t>();
    const auto latent_patch = cfg.at("latent_patch_size").get<size_t>();
    const size_t vae_downsample = size_t{1} << (vae.at("channel_multiplier").size() - 1);
    // <|im_start|> / <|im_end|> wrap every prompt; the converted checkpoint stores them as bos / eos
    const auto generation = read_json(models_dir / "generation_config.json");
    c = {hidden,
         text.at("num_hidden_layers").get<size_t>(),
         text.at("num_key_value_heads").get<size_t>(),
         head_dim,
         vae.at("latent_channels").get<size_t>(),
         latent_patch,
         vae_downsample * latent_patch,
         vision.at("patch_size").get<size_t>(),
         vision.at("max_num_patches_per_side").get<size_t>(),
         cfg.at("max_latent_size").get<size_t>(),
         generation.at("bos_token_id").get<int64_t>(),
         generation.at("eos_token_id").get<int64_t>(),
         cfg.at("start_of_image_token_id").get<int64_t>(),
         cfg.at("end_of_image_token_id").get<int64_t>()};
    read_image_processor(models_dir, c);
    return c;
}

}  // namespace

class BagelPipeline::Impl {
public:
    Impl(const std::filesystem::path& models_dir,
         const std::string& device,
         const std::filesystem::path& tokenizer_dir,
         const ov::AnyMap& properties)
        : m_npu(device.rfind("NPU", 0) == 0),
          m_tokenizer(tokenizer_dir.empty() ? models_dir : tokenizer_dir) {
        const auto cfg = read_bagel_config(models_dir);
        m_hidden = cfg.hidden;
        m_layers = cfg.layers;
        m_kv_heads = cfg.kv_heads;
        m_head_dim = cfg.head_dim;
        m_latent_channel = cfg.latent_channel;
        m_latent_patch = cfg.latent_patch;
        m_latent_downsample = cfg.latent_downsample;
        m_vit_patch = cfg.vit_patch;
        m_vit_max_side = cfg.vit_max_side;
        m_max_latent = cfg.max_latent;
        m_bos = cfg.bos;
        m_eos = cfg.eos;
        m_soi = cfg.soi;
        m_eoi = cfg.eoi;
        m_vit_transform = cfg.vit_transform;
        m_vae_transform = cfg.vae_transform;

        auto& core = utils::singleton_core();
        if (m_npu) {
            m_npu_variants = bagel::npu::VariantNames(core, device, properties);
        }
        const auto compile = [&](const std::string& name, const std::string& dev, const ov::AnyMap& config) {
            const auto t0 = Clock::now();
            auto compiled = core.compile_model((models_dir / model_file(models_dir, name)).string(), dev, config);
            m_perf.compile_ms.emplace_back("compile_model " + name.substr(0, name.find("_model.xml")), ms_since(t0));
            return compiled.create_infer_request();
        };
        // the text embedding lookup stays on CPU on NPU, like in the GenAI VLMs
        const auto config = m_npu ? bagel::npu::compile_properties(properties) : properties;
        m_text_emb = m_npu ? compile("text_embeddings_model.xml", "CPU", {})
                           : compile("text_embeddings_model.xml", device, properties);
        m_vit = compile("vision_embeddings_model.xml", device, config);
        ov::AnyMap lm_config = config;
        if (device.rfind("GPU", 0) == 0) {
            // BAGEL's outlier activations break GPU dynamic quantization of compressed weights (INT4 rel err 0.4)
            lm_config.emplace(ov::hint::dynamic_quantization_group_size.name(), uint64_t{0});
        }
        m_lm = compile("language_model.xml", device, lm_config);
        m_vae_enc = compile("vae_encoder_model.xml", device, config);
        m_vae_dec = compile("vae_decoder_model.xml", device, config);
    }

    std::string generate_text(const std::string& prompt,
                              const std::optional<ov::Tensor>& image,
                              size_t max_new_tokens,
                              const std::string& system_prompt) {
        reset();
        if (!system_prompt.empty()) {
            prefill_text(system_prompt);
        }
        if (image) {
            prefill_image(to_clip_image(*image));
        }
        prefill_text(prompt);

        int64_t cur = m_bos;
        std::vector<int64_t> out;
        for (size_t i = 0; i < max_new_tokens; ++i) {
            const auto logits = lm(embed({cur}), {m_ctx.rope}, full_mask(1, m_ctx.len), true, "decode");
            ++m_ctx.len;
            ++m_ctx.rope;
            const size_t vocab = logits.get_shape().back();
            const float* last = logits.data<const float>() + logits.get_size() - vocab;
            cur = std::max_element(last, last + vocab) - last;
            if (cur == m_eos) {
                break;
            }
            out.push_back(cur);
        }
        return m_tokenizer.decode(out, {{"skip_special_tokens", false}});
    }

    ov::Tensor generate_image(const std::string& prompt, const BagelImageGenerationConfig& config) {
        reset();
        prefill_text(prompt);
        const Context main = m_ctx;
        std::optional<Context> cfg_text;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = Context{empty_kv(), 0, 0};
        }
        const size_t side = config.size / m_latent_downsample;
        OPENVINO_ASSERT(side > 0, "size must be at least ", m_latent_downsample);
        return diffuse(side, side, main, cfg_text, std::nullopt, config, {0.4f, 1.0f});
    }

    ov::Tensor edit_image(const ov::Tensor& image,
                          const std::string& prompt,
                          const BagelImageGenerationConfig& config) {
        // as BagelImageProcessor, the ViT sees the image after the VAE resize
        const auto vae_image = m_vae_transform.resize(to_clip_image(image));
        const auto [vae_ctx, hw] = vae_prefill(vae_image, Context{empty_kv(), 0, 0});

        reset();
        m_ctx = vae_ctx;
        prefill_image(vae_image);
        const Context img_ctx = m_ctx;
        prefill_text(prompt);
        const Context main = m_ctx;

        std::optional<Context> cfg_text, cfg_img;
        if (config.cfg_text_scale > 1.0f) {
            cfg_text = img_ctx;
        }
        if (config.cfg_img_scale > 1.0f) {
            reset();
            prefill_text(prompt);
            cfg_img = m_ctx;
        }
        return diffuse(hw.first, hw.second, main, cfg_text, cfg_img, config, {0.0f, 1.0f});
    }

    BagelPerfMetrics m_perf;

private:
    // [layer * 2 + {0: key, 1: value}], each [1, kv_heads, len, head_dim]; tensors are never written after creation
    using KVCache = std::vector<ov::Tensor>;

    struct Context {
        KVCache kv;
        int64_t rope = 0;
        size_t len = 0;
    };

    struct LmOutputs {
        ov::Tensor logits;
        ov::Tensor v_t;
        KVCache present;
    };

    KVCache empty_kv() const {
        KVCache kv;
        for (size_t i = 0; i < m_layers * 2; ++i) {
            kv.emplace_back(ov::element::f32, ov::Shape{1, m_kv_heads, 0, m_head_dim});
        }
        return kv;
    }

    void reset() {
        m_ctx = Context{empty_kv(), 0, 0};
    }

    void record(const std::string& stage, Clock::time_point t0) {
        m_perf.infer_ms.emplace_back(stage, ms_since(t0));
    }

    // A not-yet-compiled NPU variant runs twice so compile and inference times split
    template <typename Fn>
    void timed(const std::string& stage, const std::string& variant, Fn&& fn) {
        if (m_npu && m_compiled.insert(variant).second) {
            auto t0 = Clock::now();
            fn();
            const float first = ms_since(t0);
            m_perf.first_call_ms += first;
            t0 = Clock::now();
            fn();
            const float infer = ms_since(t0);
            m_perf.compile_ms.emplace_back(variant, first - infer);
            m_perf.infer_ms.emplace_back(stage, infer);
            return;
        }
        const auto t0 = Clock::now();
        fn();
        record(stage, t0);
    }

    // [n, hidden] f32
    ov::Tensor embed(const std::vector<int64_t>& ids) {
        const auto t0 = Clock::now();
        m_text_emb.set_input_tensor(i64_tensor(ids));
        m_text_emb.infer();
        auto out = copy_tensor(m_text_emb.get_output_tensor());
        record("text_emb", t0);
        return out;
    }

    LmOutputs forward(const ov::Tensor& embeds,
                      const std::vector<int64_t>& und_idx,
                      const std::vector<int64_t>& gen_idx,
                      const ov::Tensor& x_t,
                      float timestep,
                      const std::vector<int64_t>& vae_pos,
                      const ov::Tensor& mask,
                      const std::vector<int64_t>& pos,
                      const KVCache& past,
                      bool need_logits,
                      bool need_kv,
                      const std::string& stage) {
        ov::Tensor ts(ov::element::f32, {1});
        ts.data<float>()[0] = timestep;
        m_lm.set_tensor("inputs_embeds", embeds);
        m_lm.set_tensor("und_index", i64_tensor(und_idx));
        m_lm.set_tensor("gen_index", i64_tensor(gen_idx));
        m_lm.set_tensor("x_t", x_t);
        m_lm.set_tensor("timestep", ts);
        m_lm.set_tensor("vae_position_ids", i64_tensor(vae_pos));
        m_lm.set_tensor("attention_mask", mask);
        m_lm.set_tensor("position_ids", i64_tensor(pos));
        for (size_t i = 0; i < m_layers; ++i) {
            m_lm.set_tensor("past_key_values." + std::to_string(i) + ".key", past[i * 2]);
            m_lm.set_tensor("past_key_values." + std::to_string(i) + ".value", past[i * 2 + 1]);
        }

        // Only the needed outputs are read: NPUW_BAGEL crops an output on first access
        timed("lm." + stage + ".infer",
              m_npu_variants.language(embeds.get_shape()[0], gen_idx.size(), past[0].get_shape()[2]),
              [&] {
                  m_lm.start_async();
                  m_lm.wait();
              });

        const auto t0 = Clock::now();
        LmOutputs out;
        if (need_kv) {
            for (size_t i = 0; i < m_layers; ++i) {
                out.present.push_back(copy_tensor(m_lm.get_tensor("present." + std::to_string(i) + ".key")));
                out.present.push_back(copy_tensor(m_lm.get_tensor("present." + std::to_string(i) + ".value")));
            }
        }
        if (need_logits) {
            out.logits = copy_tensor(m_lm.get_tensor("logits"));
        }
        out.v_t = copy_tensor(m_lm.get_tensor("v_t"));
        record("lm." + stage + ".outputs", t0);
        return out;
    }

    // Understanding-only call on the current context; updates its KV cache
    ov::Tensor lm(const ov::Tensor& embeds,
                  const std::vector<int64_t>& pos,
                  const ov::Tensor& mask,
                  bool need_logits,
                  const std::string& stage) {
        const auto s = static_cast<int64_t>(embeds.get_shape()[0]);
        auto out = forward(embeds,
                           iota_i64(0, s),
                           {},
                           ov::Tensor(ov::element::f32, {0, latent_dim()}),
                           0.0f,
                           {},
                           mask,
                           pos,
                           m_ctx.kv,
                           need_logits,
                           true,
                           stage);
        m_ctx.kv = std::move(out.present);
        return out.logits;
    }

    // Generation call: [soi, N latent tokens, eoi] on top of the given context
    LmOutputs gen(const ov::Tensor& x_t,
                  float t,
                  const Context& ctx,
                  const std::vector<int64_t>& vae_pos,
                  bool need_kv,
                  const std::string& stage) {
        const size_t n = x_t.get_shape()[0];
        const size_t s = n + 2;
        ov::Tensor embeds(ov::element::f32, {s, m_hidden});
        float* e = embeds.data<float>();
        std::fill_n(e, embeds.get_size(), 0.0f);
        std::copy_n(m_soi_emb.data<const float>(), m_hidden, e);
        std::copy_n(m_eoi_emb.data<const float>(), m_hidden, e + (s - 1) * m_hidden);
        return forward(embeds,
                       {0, static_cast<int64_t>(s - 1)},
                       iota_i64(1, static_cast<int64_t>(s - 1)),
                       x_t,
                       t,
                       vae_pos,
                       full_mask(s, ctx.len),
                       std::vector<int64_t>(s, ctx.rope),
                       ctx.kv,
                       false,
                       need_kv,
                       stage);
    }

    void load_soi_eoi() {
        m_soi_emb = embed({m_soi});
        m_eoi_emb = embed({m_eoi});
    }

    void prefill_text(const std::string& text) {
        const auto tokenized = m_tokenizer.encode(text).input_ids;
        const int64_t* tok = tokenized.data<const int64_t>();
        std::vector<int64_t> ids{m_bos};
        ids.insert(ids.end(), tok, tok + tokenized.get_size());
        ids.push_back(m_eos);
        const size_t t = ids.size();
        lm(embed(ids),
           iota_i64(m_ctx.rope, m_ctx.rope + static_cast<int64_t>(t)),
           causal_mask(t, m_ctx.len),
           false,
           "text_prefill");
        m_ctx.len += t;
        m_ctx.rope += static_cast<int64_t>(t);
    }

    void prefill_image(const clip_image_u8& image) {
        const auto img = to_normalized_tensor(m_vit_transform.resize(image));
        const size_t h = img.get_shape()[1], w = img.get_shape()[2];
        const size_t gh = h / m_vit_patch, gw = w / m_vit_patch;
        const auto patches = patchify(img.data<const float>(), 3, h, w, m_vit_patch, gh, gw);
        const auto pos = i64_tensor(flat_position_ids(gh, gw, m_vit_max_side));

        timed("vit", "vision " + std::to_string(gh * gw) + " patches", [&] {
            m_vit.set_tensor("pixel_values", patches);
            m_vit.set_tensor("position_ids", pos);
            m_vit.infer();
        });
        const auto img_emb = m_vit.get_output_tensor();

        load_soi_eoi();
        const size_t n = img_emb.get_shape()[0];
        const size_t s = n + 2;
        ov::Tensor seq(ov::element::f32, {s, m_hidden});
        float* d = seq.data<float>();
        std::copy_n(m_soi_emb.data<const float>(), m_hidden, d);
        std::copy_n(img_emb.data<const float>(), n * m_hidden, d + m_hidden);
        std::copy_n(m_eoi_emb.data<const float>(), m_hidden, d + (s - 1) * m_hidden);
        lm(seq, std::vector<int64_t>(s, m_ctx.rope), full_mask(s, m_ctx.len), false, "image_prefill");
        m_ctx.len += s;
        m_ctx.rope += 1;
    }

    size_t latent_dim() const {
        return m_latent_patch * m_latent_patch * m_latent_channel;
    }

    std::vector<int64_t> vae_position_ids(size_t h, size_t w) const {
        return flat_position_ids(h, w, m_max_latent);
    }

    // Encodes the VAE-resized image into a [soi, latent, eoi] context; returns it with the latent grid
    std::pair<Context, std::pair<size_t, size_t>> vae_prefill(const clip_image_u8& vae_image, const Context& ctx) {
        const auto img = to_normalized_tensor(vae_image);
        const size_t img_h = img.get_shape()[1], img_w = img.get_shape()[2];
        ov::Tensor batched(ov::element::f32, {1, 3, img_h, img_w}, img.data());
        timed("vae_enc", "vae_encoder " + std::to_string(img_h) + "x" + std::to_string(img_w), [&] {
            m_vae_enc.set_input_tensor(batched);
            m_vae_enc.infer();
        });
        const auto latent = m_vae_enc.get_output_tensor();
        const auto& ls = latent.get_shape();
        const size_t h = img_h / m_latent_downsample, w = img_w / m_latent_downsample;
        const auto x_t = patchify(latent.data<const float>(), m_latent_channel, ls[2], ls[3], m_latent_patch, h, w);

        load_soi_eoi();
        auto out = gen(x_t, 0.0f, ctx, vae_position_ids(h, w), true, "vae_prefill");
        return {Context{std::move(out.present), ctx.rope + 1, ctx.len + h * w + 2}, {h, w}};
    }

    std::vector<float> velocity(const ov::Tensor& x_t,
                                float t,
                                const Context& cond,
                                const std::vector<int64_t>& vae_pos) {
        const auto v = gen(x_t, t, cond, vae_pos, false, "denoise").v_t;
        return {v.data<const float>(), v.data<const float>() + v.get_size()};
    }

    ov::Tensor diffuse(size_t h,
                       size_t w,
                       const Context& main,
                       const std::optional<Context>& cfg_text,
                       const std::optional<Context>& cfg_img,
                       const BagelImageGenerationConfig& config,
                       std::pair<float, float> default_interval) {
        const size_t n = h * w;
        const size_t dim = latent_dim();
        const auto vae_pos = vae_position_ids(h, w);
        const auto interval = config.cfg_interval.value_or(default_interval);
        load_soi_eoi();

        ov::Tensor x(ov::element::f32, {n, dim});
        float* xd = x.data<float>();
        NumpyRandomState rng(config.seed);
        for (size_t i = 0; i < n * dim; ++i) {
            xd[i] = static_cast<float>(rng.gauss());
        }

        // np.linspace(1, 0, steps, dtype=float32) with the timestep shift applied in f32
        const size_t steps = config.num_inference_steps;
        OPENVINO_ASSERT(steps >= 2, "num_inference_steps must be at least 2");
        std::vector<float> ts(steps);
        for (size_t i = 0; i < steps; ++i) {
            ts[i] = i + 1 == steps ? 0.0f : static_cast<float>(i * (-1.0 / (steps - 1)) + 1.0);
            const float shift = config.timestep_shift;
            ts[i] = shift * ts[i] / (1.0f + (shift - 1.0f) * ts[i]);
        }

        for (size_t i = 0; i + 1 < steps; ++i) {
            const auto t0 = Clock::now();
            const float first0 = m_perf.first_call_ms;
            const float t = ts[i];
            const bool use_cfg = interval.first < t && t <= interval.second;
            auto v = velocity(x, t, main, vae_pos);
            if (use_cfg) {
                auto v_ = v;
                if (cfg_text && config.cfg_text_scale > 1.0f) {
                    const auto vt = velocity(x, t, *cfg_text, vae_pos);
                    for (size_t k = 0; k < v.size(); ++k) {
                        v_[k] = vt[k] + config.cfg_text_scale * (v[k] - vt[k]);
                    }
                }
                if (cfg_img && config.cfg_img_scale > 1.0f) {
                    const auto vi = velocity(x, t, *cfg_img, vae_pos);
                    for (size_t k = 0; k < v.size(); ++k) {
                        v_[k] = vi[k] + config.cfg_img_scale * (v_[k] - vi[k]);
                    }
                }
                const float scale = std::max(std::min(l2_norm(v) / (l2_norm(v_) + 1e-8f), 1.0f), config.cfg_renorm_min);
                for (size_t k = 0; k < v.size(); ++k) {
                    v[k] = v_[k] * scale;
                }
            }
            const float dt = ts[i] - ts[i + 1];
            for (size_t k = 0; k < v.size(); ++k) {
                xd[k] = xd[k] - v[k] * dt;
            }
            m_perf.infer_ms.emplace_back(use_cfg ? "step.cfg" : "step.nocfg",
                                         ms_since(t0) - (m_perf.first_call_ms - first0));
        }
        return decode(x, h, w);
    }

    // [n, p * p * c] latent tokens -> u8 [1, H, W, 3] image
    ov::Tensor decode(const ov::Tensor& x, size_t h, size_t w) {
        const size_t p = m_latent_patch, c = m_latent_channel;
        const size_t lh = h * p, lw = w * p;
        ov::Tensor lat(ov::element::f32, {1, c, lh, lw});
        const float* src = x.data<const float>();
        float* dst = lat.data<float>();
        for (size_t gh = 0; gh < h; ++gh) {
            for (size_t gw = 0; gw < w; ++gw) {
                for (size_t py = 0; py < p; ++py) {
                    for (size_t px = 0; px < p; ++px) {
                        for (size_t ch = 0; ch < c; ++ch) {
                            dst[(ch * lh + gh * p + py) * lw + gw * p + px] = *src++;
                        }
                    }
                }
            }
        }
        timed("vae_dec", "vae_decoder " + std::to_string(lh) + "x" + std::to_string(lw), [&] {
            m_vae_dec.set_input_tensor(lat);
            m_vae_dec.infer();
        });
        const auto img = m_vae_dec.get_output_tensor();
        const size_t oh = img.get_shape()[2], ow = img.get_shape()[3];
        ov::Tensor out(ov::element::u8, {1, oh, ow, 3});
        const float* in = img.data<const float>();
        uint8_t* o = out.data<uint8_t>();
        for (size_t i = 0; i < oh * ow; ++i) {
            for (size_t ch = 0; ch < 3; ++ch) {
                const float v = std::clamp((in[ch * oh * ow + i] + 1.0f) / 2.0f, 0.0f, 1.0f);
                o[i * 3 + ch] = static_cast<uint8_t>(v * 255.0f);
            }
        }
        return out;
    }

    bool m_npu;
    Tokenizer m_tokenizer;
    ov::InferRequest m_text_emb, m_vit, m_lm, m_vae_enc, m_vae_dec;
    std::set<std::string> m_compiled;
    bagel::npu::VariantNames m_npu_variants;
    ImageTransform m_vit_transform{}, m_vae_transform{};
    Context m_ctx;
    ov::Tensor m_soi_emb, m_eoi_emb;

    size_t m_hidden = 0, m_layers = 0, m_kv_heads = 0, m_head_dim = 0;
    size_t m_latent_channel = 0, m_latent_patch = 0, m_latent_downsample = 0;
    size_t m_vit_patch = 0, m_vit_max_side = 0, m_max_latent = 0;
    int64_t m_bos = 0, m_eos = 0, m_soi = 0, m_eoi = 0;
};

BagelPipeline::BagelPipeline(const std::filesystem::path& models_dir,
                             const std::string& device,
                             const std::filesystem::path& tokenizer_dir,
                             const ov::AnyMap& properties)
    : m_impl(std::make_unique<Impl>(models_dir, device, tokenizer_dir, properties)) {}

BagelPipeline::~BagelPipeline() = default;

std::string BagelPipeline::generate_text(const std::string& prompt,
                                         const std::optional<ov::Tensor>& image,
                                         size_t max_new_tokens,
                                         const std::string& system_prompt) {
    return m_impl->generate_text(prompt, image, max_new_tokens, system_prompt);
}

ov::Tensor BagelPipeline::generate_image(const std::string& prompt, const BagelImageGenerationConfig& config) {
    return m_impl->generate_image(prompt, config);
}

ov::Tensor BagelPipeline::edit_image(const ov::Tensor& image,
                                     const std::string& prompt,
                                     const BagelImageGenerationConfig& config) {
    return m_impl->edit_image(image, prompt, config);
}

const BagelPerfMetrics& BagelPipeline::get_perf_metrics() const {
    return m_impl->m_perf;
}

void BagelPipeline::reset_perf_metrics() {
    m_impl->m_perf = {};
}

}  // namespace ov::genai
