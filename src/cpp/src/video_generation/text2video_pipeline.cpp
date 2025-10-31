// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"
#include "image_generation/schedulers/ischeduler.hpp"
#include "image_generation/numpy_utils.hpp"
#include <openvino/op/transpose.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

#include <openvino/op/transpose.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/round.hpp>

using namespace ov::genai;

namespace {

VideoGenerationConfig LTX_VIDEO_DEFAULT_CONFIG = VideoGenerationConfig{
    ImageGenerationConfig{
        .guidance_scale = 3.0f,
        .height = 512,
        .width = 704,
        .num_inference_steps = 50,
        .max_sequence_length = 128,
        .strength = 1.0f,
    },
    .guidance_rescale = 0.0,
    .num_frames = 161,
    .frame_rate = 25.0f
};

// Some defaults aren't special values so it's not possible to distinguish
// whether user set them or not. Replace only special values.
void replace_defaults(VideoGenerationConfig& config) {
    if (-1 == config.height) {
        config.height = LTX_VIDEO_DEFAULT_CONFIG.height;
    }
    if (-1 == config.width) {
        config.width = LTX_VIDEO_DEFAULT_CONFIG.width;
    }
    if (-1 == config.max_sequence_length) {
        config.max_sequence_length = LTX_VIDEO_DEFAULT_CONFIG.max_sequence_length;
    }
    if (std::isnan(config.guidance_rescale)) {
        config.guidance_rescale = LTX_VIDEO_DEFAULT_CONFIG.guidance_rescale;
    }
    if (0 == config.num_frames) {
        config.num_frames = LTX_VIDEO_DEFAULT_CONFIG.num_frames;
    }
    if (std::isnan(config.frame_rate)) {
        config.frame_rate = LTX_VIDEO_DEFAULT_CONFIG.frame_rate;
    }
}

std::shared_ptr<IScheduler> cast_scheduler(std::shared_ptr<Scheduler>&& scheduler) {
    auto casted = std::dynamic_pointer_cast<IScheduler>(std::move(scheduler));
    OPENVINO_ASSERT(casted != nullptr, "Passed incorrect scheduler type");
    return casted;
}

void check_inputs(const VideoGenerationConfig& generation_config, size_t vae_scale_factor) {
    generation_config.validate();
    OPENVINO_ASSERT(generation_config.height > 0, "Height must be positive");
    OPENVINO_ASSERT(generation_config.height % 32 == 0, "Height have to be divisible by 32 but got ", generation_config.height);
    OPENVINO_ASSERT(generation_config.width > 0, "Width must be positive");
    OPENVINO_ASSERT(generation_config.width % 32 == 0, "Width have to be divisible by 32 but got ", generation_config.width);
    OPENVINO_ASSERT(1.0f == generation_config.strength, "Strength isn't applicable. Must be set to the default 1.0");

    OPENVINO_ASSERT(!generation_config.prompt_2.has_value(), "Prompt 2 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.prompt_3.has_value(), "Prompt 3 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.negative_prompt_2.has_value(), "Negative prompt 2 is not used by LTXPipeline.");
    OPENVINO_ASSERT(!generation_config.negative_prompt_3.has_value(), "Negative prompt 3 is not used by LTXPipeline.");
    OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");
    OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
    OPENVINO_ASSERT(
        (generation_config.height % vae_scale_factor == 0 || generation_config.height < 0)
            && (generation_config.width % vae_scale_factor == 0 || generation_config.width < 0),
        "Both 'width' and 'height' must be divisible by ",
        vae_scale_factor
    );
}

// Unpacked latents of shape [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
// The patch dimensions are then permuted and collapsed into the channel dimension of shape:
// [B, F // p_t * H // p * W // p, C * p_t * p * p] (a 3 dimensional tensor).
// dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
ov::Tensor pack_latents(ov::Tensor& latents, size_t patch_size, size_t patch_size_t) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape.at(0), num_channels = latents_shape.at(1), num_frames = latents_shape.at(2), height = latents_shape.at(3), width = latents_shape.at(4);
    size_t post_patch_num_frames = num_frames / patch_size_t;
    size_t post_patch_height = height / patch_size;
    size_t post_patch_width = width / patch_size;
    latents.set_shape({batch_size, num_channels, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size});
    std::array<int64_t, 8> order = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(ov::element::f32, {})};
    ov::op::v1::Transpose{}.evaluate(outputs, {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, order.data())});
    ov::Shape permuted_shape = outputs.at(0).get_shape();
    outputs.at(0).set_shape({permuted_shape.at(0), permuted_shape.at(1) * permuted_shape.at(2) * permuted_shape.at(3), permuted_shape.at(4) * permuted_shape.at(5) * permuted_shape.at(6)});
    return outputs.at(0);
}

// Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
// are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of what happens in the `_pack_latents` method.
ov::Tensor unpack_latents(ov::Tensor& latents,
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

    latents.set_shape({batch_size, num_frames, height, width, num_channels, patch_size_t, patch_size, patch_size});

    // permute(0, 4, 1, 5, 2, 6, 3, 7) -> [B, C, F//patch_size_t, patch_size_t, H//patch_size, patch_size, W//patch_size, patch_size]
    const std::array<int64_t, 8> order = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<ov::Tensor> outputs{ov::Tensor(latents.get_element_type(), {})};
    ov::op::v1::Transpose{}.evaluate(
        outputs,
        {latents, ov::Tensor(ov::element::i64, ov::Shape{order.size()}, const_cast<int64_t*>(order.data()))}
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
    for (auto d : t.get_shape()) elems *= d;

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
ov::Tensor denormalize_latents(ov::Tensor latents,
                               ov::Tensor latents_mean,
                               ov::Tensor latents_std,
                               float scaling_factor = 1.0f) {
    const ov::Shape latents_shape = latents.get_shape();
    OPENVINO_ASSERT(latents_shape.size() == 5, "denormalize_latents expects [B, C, F, H, W]");
    const size_t num_channels = latents_shape[1];

    // .view(1, -1, 1, 1, 1)
    reshape_to_1C111(latents_mean, num_channels);
    reshape_to_1C111(latents_std,  num_channels);

    const auto latents_type = latents.get_element_type();
    ov::Tensor scale = make_scalar(latents_type, scaling_factor);

    // latents * latents_std
    std::vector<ov::Tensor> tmp{ov::Tensor(latents_type, {})};
    ov::op::v1::Multiply{}.evaluate(tmp, {latents, latents_std}); // NUMPY broadcast

    // (...) / scaling_factor
    std::vector<ov::Tensor> tmp2{ov::Tensor(latents_type, {})};
    ov::op::v1::Divide{}.evaluate(tmp2, {tmp[0], scale});

    // (...) + latents_mean
    std::vector<ov::Tensor> result{ov::Tensor(latents_type, {})};
    ov::op::v1::Add{}.evaluate(result, {tmp2[0], latents_mean});

    return result[0]; // [B, C, F, H, W]
}

// For debug only
void saveTensorToFile(const ov::Tensor& tensor, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
 
    ov::Shape shape = tensor.get_shape();
    for (size_t dim : shape) {
        file << dim << " ";
    }
    file << "\n";
 
    float* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        file << data[i] << " ";
    }
    file << "\n";
    file.close();
}

// for debug only
ov::Tensor loadTensorFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("File format error: missing shape line");
    }

    std::istringstream shape_stream(line);
    std::vector<size_t> shape_vec;
    size_t dim;
    while (shape_stream >> dim) {
        shape_vec.push_back(dim);
    }
    ov::Shape shape(shape_vec);


    std::vector<float> data;
    while (std::getline(file, line)) {
        std::istringstream data_stream(line);
        float value;
        while (data_stream >> value) {
            data.push_back(value);
        }
    }

    size_t expected_size = ov::shape_size(shape);
    if (data.size() != expected_size) {
        throw std::runtime_error("Data size mismatch: expected " +
                                std::to_string(expected_size) +
                                ", but got " + std::to_string(data.size()));
    }

    ov::Tensor tensor(ov::element::f32, shape);
    std::memcpy(tensor.data<float>(), data.data(), data.size() * sizeof(float));

    return tensor;
}


ov::Tensor prepare_latents(const ov::genai::VideoGenerationConfig& generation_config, size_t num_channels_latents, size_t spatial_compression_ratio, size_t temporal_compression_ratio, size_t transformer_spatial_patch_size, size_t transformer_temporal_patch_size) {
    size_t height = generation_config.height / spatial_compression_ratio;
    size_t width = generation_config.width / spatial_compression_ratio;
    size_t latent_num_frames = (generation_config.num_frames - 1) / temporal_compression_ratio + 1;
    ov::Shape shape{generation_config.num_images_per_prompt, num_channels_latents, latent_num_frames, height, width};
    // ov::Tensor latents = generation_config.generator->randn_tensor(shape);
    ov::Tensor latents = loadTensorFromFile("../latents_before_pack.txt");
    return pack_latents(latents, transformer_spatial_patch_size, transformer_temporal_patch_size);
}
}  // anonymous namespace

void VideoGenerationConfig::validate() const {
    ImageGenerationConfig::validate();
}

void VideoGenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    ImageGenerationConfig::update_generation_config(properties);
    using ov::genai::utils::read_anymap_param;
    read_anymap_param(properties, "guidance_rescale", guidance_rescale);
    read_anymap_param(properties, "num_frames", num_frames);
    read_anymap_param(properties, "frame_rate", frame_rate);
    replace_defaults(*this);
}

class Text2VideoPipeline::LTXPipeline {
    using Ms = std::chrono::duration<float, std::ratio<1, 1000>>;

    std::shared_ptr<IScheduler> m_scheduler;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<LTXVideoTransformer3DModel> m_transformer;
    std::shared_ptr<AutoencoderKLLTXVideo> m_vae;
    VideoGenerationPerfMetrics m_perf_metrics;
    double m_latent_timestep = -1.0;  // TODO: float?
    Ms m_load_time;

    void compute_hidden_states(const std::string& positive_prompt, const std::string& negative_prompt, const VideoGenerationConfig& generation_config) {
        auto infer_start = std::chrono::steady_clock::now();
        bool do_classifier_free_guidance = generation_config.guidance_scale > 1.0;

        // torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        // genai m_t5_text_encoder->infer retuns the same tensor [negative_prompt_embeds, prompt_embeds]
        ov::Tensor prompt_embeds = m_t5_text_encoder->infer(
            positive_prompt,
            negative_prompt,
            do_classifier_free_guidance,
            generation_config.max_sequence_length, {
                ov::genai::pad_to_max_length(true),
                ov::genai::max_length(generation_config.max_sequence_length),
                ov::genai::add_special_tokens(true)
            }
        );
        ov::Tensor prompt_attention_mask = m_t5_text_encoder->get_prompt_attention_mask();
        auto infer_end = std::chrono::steady_clock::now();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = Ms{infer_end - infer_start}.count();

        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);
        prompt_attention_mask = numpy_utils::repeat(prompt_attention_mask, generation_config.num_images_per_prompt);

        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("encoder_attention_mask", prompt_attention_mask);
    }

public:
    VideoGenerationConfig m_generation_config;

    // for debug only
    void print_ov_tensor(const ov::Tensor& tensor, const std::string& name = "") {
        if (!name.empty())
            std::cout << name << ": ";

        std::cout << "shape = [";
        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
            std::cout << tensor.get_shape()[i];
            if (i + 1 < tensor.get_shape().size())
                std::cout << ", ";
        }
        std::cout << "], type = " << tensor.get_element_type() << ", values = [";

        const size_t limit = 10;
        const size_t size = tensor.get_size();

        if (tensor.get_element_type() == ov::element::f32) {
            const float* data = tensor.data<const float>();
            for (size_t i = 0; i < std::min(size, limit); ++i) {
                std::cout << std::fixed << std::setprecision(4) << data[i];
                if (i + 1 < std::min(size, limit))
                    std::cout << ", ";
            }
        } else if (tensor.get_element_type() == ov::element::i64) {
            const int64_t* data = tensor.data<const int64_t>();
            for (size_t i = 0; i < std::min(size, limit); ++i) {
                std::cout << data[i];
                if (i + 1 < std::min(size, limit))
                    std::cout << ", ";
            }
        } 
        else if (tensor.get_element_type() == ov::element::u8) {
            const uint8_t* data = tensor.data<const uint8_t>();
            for (size_t i = 0; i < std::min(size, limit); ++i) {
                std::cout << static_cast<int>(data[i]);
                if (i + 1 < std::min(size, limit))
                    std::cout << ", ";
            }
        } else {
            std::cout << "<unsupported type>";
        }

        if (size > limit)
            std::cout << ", ...";

        std::cout << "]" << std::endl;
    }

    LTXPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties,
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()
    ) :
            m_scheduler{cast_scheduler(Scheduler::from_config(models_dir / "scheduler/scheduler_config.json"))},
            m_t5_text_encoder{std::make_shared<T5EncoderModel>(models_dir / "text_encoder", device, properties)},
            m_transformer{std::make_shared<LTXVideoTransformer3DModel>(models_dir / "transformer", device, properties)},
            m_vae{std::make_shared<AutoencoderKLLTXVideo>(models_dir / "vae_decoder", device, properties)},
            m_generation_config{LTX_VIDEO_DEFAULT_CONFIG} {
        const std::filesystem::path model_index_path = models_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);
        OPENVINO_ASSERT("LTXPipeline" == nlohmann::json::parse(file)["_class_name"].get<std::string>());
        m_load_time = Ms{std::chrono::steady_clock::now() - start_time};
    }

    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0;
    }

    // x = (images * 0.5 + 0.5).clamp(0, 1)
    inline ov::Tensor denormalize_01(const ov::Tensor& images) {
        const auto et = images.get_element_type();
        const auto shape = images.get_shape();

        std::vector<ov::Tensor> out{ov::Tensor(et, shape)};
        std::vector<ov::Tensor> in;

        // x = images * 0.5
        in = {images, make_scalar(et, 0.5f)};
        ov::op::v1::Multiply{}.evaluate(out, in);

        // x = x + 0.5
        in = {out[0], make_scalar(et, 0.5f)};
        ov::op::v1::Add{}.evaluate(out, in);

        // x = min(x, 1.0)
        in = {out[0], make_scalar(et, 1.0f)};
        ov::op::v1::Minimum{}.evaluate(out, in);

        // x = max(x, 0.0)
        in = {out[0], make_scalar(et, 0.0f)};
        ov::op::v1::Maximum{}.evaluate(out, in);

        return out[0];
    }

    // inp:  [B, C, F, H, W], out: [B, F, H, W, C] uint8
    inline ov::Tensor postprocess_video(const ov::Tensor& video) {
        const auto& shape = video.get_shape();
        OPENVINO_ASSERT(shape.size() == 5, "postprocess_video expects [B, C, F, H, W]");
        const auto et = video.get_element_type();

        // [B, C, F, H, W] to [B, F, H, W, C]
        const std::array<int64_t, 5> order = {0, 2, 3, 4, 1};
        ov::Tensor perm_order(ov::element::i64, {order.size()}, const_cast<int64_t*>(order.data()));

        std::vector<ov::Tensor> trans_out{ov::Tensor(et, shape)};
        ov::op::v1::Transpose{}.evaluate(trans_out, {video, perm_order});
        ov::Tensor transposed = trans_out[0];

        // (x * 0.5 + 0.5).clamp(0, 1)
        ov::Tensor normalized = denormalize_01(transposed);

        // (x * 255)
        std::vector<ov::Tensor> scaled{ov::Tensor(et, normalized.get_shape())};
        ov::op::v1::Multiply{}.evaluate(scaled, {normalized, make_scalar(et, 255.0f)});

        // round()
        std::vector<ov::Tensor> rounded{ov::Tensor(et, scaled[0].get_shape())};
        ov::op::v5::Round{}.evaluate(rounded, {scaled[0]});

        // to uint8
        std::vector<ov::Tensor> out_vec{ov::Tensor(ov::element::u8, rounded[0].get_shape())};
        ov::op::v0::Convert{}.evaluate(out_vec, {rounded[0]});

        return out_vec[0];  // [B, F, H, W, C]
}

    ov::Tensor generate(const std::string& positive_prompt, const std::string& negative_prompt, const ov::AnyMap& properties = {}) {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        VideoGenerationConfig merged_generation_config = m_generation_config;
        merged_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(merged_generation_config.guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG


        check_inputs(merged_generation_config, vae_scale_factor);

        // Use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        compute_hidden_states(positive_prompt, negative_prompt, merged_generation_config);

        size_t num_channels_latents = m_transformer->get_config().in_channels;
        size_t spatial_compression_ratio = m_vae->get_config().patch_size * std::pow(2, std::reduce(m_vae->get_config().spatio_temporal_scaling.begin(), m_vae->get_config().spatio_temporal_scaling.end(), 0));
        size_t temporal_compression_ratio = m_vae->get_config().patch_size_t * std::pow(2, std::reduce(m_vae->get_config().spatio_temporal_scaling.begin(), m_vae->get_config().spatio_temporal_scaling.end(), 0));
        size_t transformer_spatial_patch_size = m_transformer->get_config().patch_size;
        size_t transformer_temporal_patch_size = m_transformer->get_config().patch_size_t;

        ov::Tensor latent = prepare_latents(
            merged_generation_config,
            num_channels_latents,
            spatial_compression_ratio,
            temporal_compression_ratio,
            transformer_spatial_patch_size,
            transformer_temporal_patch_size
        );

        // Prepare timesteps
        size_t latent_num_frames = (merged_generation_config.num_frames - 1) / temporal_compression_ratio + 1;
        size_t latent_height = merged_generation_config.height / spatial_compression_ratio;  // TODO: prepare_latents() does the same
        size_t latent_width = merged_generation_config.width / spatial_compression_ratio;
        size_t video_sequence_length = latent_num_frames * latent_height * latent_width;
        m_scheduler->set_timesteps(video_sequence_length, merged_generation_config.num_inference_steps, merged_generation_config.strength);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        // Prepare micro-conditions
        //TODO: move to compute_hidden_states
        ov::Tensor rope_interpolation_scale(ov::element::f32, {3});
        rope_interpolation_scale.data<float>()[0] = static_cast<float>(temporal_compression_ratio) / merged_generation_config.frame_rate;
        rope_interpolation_scale.data<float>()[1] = spatial_compression_ratio;
        rope_interpolation_scale.data<float>()[2] = spatial_compression_ratio;
        m_transformer->set_hidden_states("rope_interpolation_scale", rope_interpolation_scale);
        print_ov_tensor(rope_interpolation_scale, "rope_interpolation_scale");

        auto make_scalar_tensor = [](size_t value) {
            ov::Tensor scalar(ov::element::i64, {});
            scalar.data<int64_t>()[0] = value;
            return scalar;
        };
        m_transformer->set_hidden_states("num_frames", make_scalar_tensor(latent_num_frames));
        m_transformer->set_hidden_states("height", make_scalar_tensor(latent_height));
        m_transformer->set_hidden_states("width", make_scalar_tensor(latent_width));

        // // Prepare timesteps
        ov::Tensor timestep(ov::element::f32, {2});
        float* timestep_data = timestep.data<float>();

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        // Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, 0, merged_generation_config.num_images_per_prompt);
                numpy_utils::batch_copy(latent, latent_cfg, 0, merged_generation_config.num_images_per_prompt, merged_generation_config.num_images_per_prompt);
            } else {
                // just assign to save memory copy
                latent_cfg = latent;
            }

            timestep_data[0] = timesteps[inference_step];
            timestep_data[1] = timesteps[inference_step];

            auto infer_start = std::chrono::steady_clock::now();
            print_ov_tensor(latent_cfg, "latent input");
            saveTensorToFile(latent_cfg, "latent_cfg_" + std::to_string(timestep_data[0]) +".txt");
            saveTensorToFile(timestep, "timestep_" + std::to_string(timestep_data[0]) +".txt");

            std::cout << "timestep " << timestep_data[0] << std::endl;
            ov::Tensor noise_pred_tensor = m_transformer->infer(latent_cfg, timestep);
            print_ov_tensor(noise_pred_tensor, "noise_pred");
            // saveTensorToFile(noise_pred_tensor, "noise_pred_tensor_" + std::to_string(timestep_data[0]) +".txt");
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;

            if (batch_size_multiplier > 1) {
                noisy_residual_tensor.set_shape(noise_pred_shape);

                // perform guidance
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] +
                                        merged_generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            // print_ov_tensor(noisy_residual_tensor, "noise_pred 2");

            print_ov_tensor(latent, "latents before step");

            // saveTensorToFile(latent, "latents_before_step_" + std::to_string(timestep_data[0]) +".txt");

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, merged_generation_config.generator);
            latent = scheduler_step_result["latent"];

            // saveTensorToFile(latent, "latents_after_step_" + std::to_string(timestep_data[0]) +".txt");

            print_ov_tensor(latent, "latents after step");
        }

        // latent = loadTensorFromFile("/home/alikh/projects/openvino.genai/transformer_output.txt");

        latent = unpack_latents(latent,
                                latent_num_frames,
                                latent_height,
                                latent_width,
                                transformer_spatial_patch_size,
                                transformer_temporal_patch_size);
        print_ov_tensor(latent, "unpack_latents");

        auto tensor_from_vector = [](const std::vector<float>& data) -> ov::Tensor {
            ov::Tensor t{ov::element::f32, ov::Shape{data.size()}};
            if (!data.empty()) {
                std::memcpy(t.data<float>(), data.data(), data.size() * sizeof(float));
            }
            return t;
        };

        latent = denormalize_latents(latent,
                                    tensor_from_vector(m_vae->get_config().latents_mean_data),
                                    tensor_from_vector(m_vae->get_config().latents_std_data),
                                    m_vae->get_config().scaling_factor);
        print_ov_tensor(latent, "denormalize_latents");
        saveTensorToFile(latent, "denormalize_latents.txt");

        // TODO: if not self.vae.config.timestep_conditioning: ... else: ...

        const auto decode_start = std::chrono::steady_clock::now();
        ov::Tensor video = m_vae->decode(latent);
        print_ov_tensor(video, "video");
        m_perf_metrics.vae_decoder_inference_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
                .count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        video = postprocess_video(video);
        print_ov_tensor(video, "postprocess_video");

        //     if (callback && callback(inference_step, timesteps.size(), latents)) {
        //         auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
        //         m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

        //         auto image = ov::Tensor(ov::element::u8, {});
        //         m_perf_metrics.generate_duration =
        //             std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start)
        //                 .count();
        //         return image;
        //     }

        //     auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
        //     m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));

        // latents = unpack_latents(latents, custom_generation_config.height, custom_generation_config.width, vae_scale_factor);
        // const auto decode_start = std::chrono::steady_clock::now();
        // auto image = m_vae->decode(latents);
        // m_perf_metrics.vae_decoder_inference_duration =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start)
        //         .count();
        // m_perf_metrics.generate_duration =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        // return image;
        return video;
    }
};

Text2VideoPipeline::Text2VideoPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const AnyMap& properties
) : m_impl{std::make_unique<ov::genai::Text2VideoPipeline::LTXPipeline>(
    models_dir, device, properties
)} {}

ov::Tensor Text2VideoPipeline::generate(
    const std::string& positive_prompt,
    const std::string& negative_prompt,
    const ov::AnyMap& properties
) {
    return m_impl->generate(positive_prompt, negative_prompt, properties);
}

const VideoGenerationConfig& Text2VideoPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void Text2VideoPipeline::set_generation_config(const VideoGenerationConfig& generation_config) {
    generation_config.validate();
    m_impl->m_generation_config = generation_config;
    replace_defaults(m_impl->m_generation_config);
}

Text2VideoPipeline::~Text2VideoPipeline() = default;

