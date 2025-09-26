// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_generation/stable_diffusion_pipeline.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"

namespace ov {
namespace genai {

class StableDiffusionXLPipeline : public StableDiffusionPipeline {
public:
    StableDiffusionXLPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        StableDiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder =
                std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(
                root_dir / "text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder_2, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet");
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE ||
                        m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder",
                                                        root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());

        // initialize force_zeros_for_empty_prompt, which is SDXL specific
        read_json_param(data, "force_zeros_for_empty_prompt", m_force_zeros_for_empty_prompt);
    }

    StableDiffusionXLPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) :
        StableDiffusionPipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const auto [properties_without_blob, blob_path] = utils::extract_export_properties(properties);

        auto updated_properties = update_adapters_in_properties(properties_without_blob, &DiffusionPipeline::derived_adapters);
        // updated_properies are for passing to the pipeline subcomponents only, not for the generation config

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            if (blob_path.has_value()) {
                updated_properties.fork()[ov::genai::blob_path.name()] = blob_path.value() / "text_encoder";
            }
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(
                root_dir / "text_encoder",
                device,
                *properties_for_text_encoder(*updated_properties, "lora_te1")
            );
            updated_properties.fork().erase(ov::genai::blob_path.name());
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            if (blob_path.has_value()) {
                updated_properties.fork()[ov::genai::blob_path.name()] = blob_path.value() / "text_encoder_2";
            }
            m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(
                root_dir / "text_encoder_2",
                device,
                *properties_for_text_encoder(*updated_properties, "lora_te2")
            );
            updated_properties.fork().erase(ov::genai::blob_path.name());
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder_2, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            if (blob_path.has_value()) {
                updated_properties.fork()[ov::genai::blob_path.name()] = blob_path.value() / "unet";
            }
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet", device, *updated_properties);
            updated_properties.fork().erase(ov::genai::blob_path.name());
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        // Temporary fix for GPU
        if (device.find("GPU") != std::string::npos &&
            updated_properties->find("INFERENCE_PRECISION_HINT") == updated_properties->end()) {
            updated_properties.fork()["WA_INFERENCE_PRECISION_HINT"] = ov::element::f32;
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (blob_path.has_value()) {
                updated_properties.fork()[ov::genai::blob_path.name()] = blob_path.value();
            }
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, *updated_properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, *updated_properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
            updated_properties.fork().erase(ov::genai::blob_path.name());
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());

        // initialize force_zeros_for_empty_prompt, which is SDXL specific
        read_json_param(data, "force_zeros_for_empty_prompt", m_force_zeros_for_empty_prompt);

        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    StableDiffusionXLPipeline(
        PipelineType pipeline_type,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae)
        : StableDiffusionPipeline(pipeline_type, clip_text_model, unet, vae) {
        m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(clip_text_model_with_projection);
        // initialize generation config
        initialize_generation_config("StableDiffusionXLPipeline");
        // here we implicitly imply that force_zeros_for_empty_prompt is set to True as by default in diffusers
        m_force_zeros_for_empty_prompt = true;
    }

    StableDiffusionXLPipeline(PipelineType pipeline_type, const StableDiffusionXLPipeline& pipe) :
        StableDiffusionXLPipeline(pipe) {
        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        m_root_dir = pipe.m_root_dir;

        m_clip_text_encoder = std::make_shared<CLIPTextModel>(*pipe.m_clip_text_encoder);
        m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(*pipe.m_clip_text_encoder_with_projection);
        m_unet = std::make_shared<UNet2DConditionModel>(*pipe.m_unet);
        m_vae = std::make_shared<AutoencoderKL>(*pipe.m_vae);

        m_pipeline_type = pipeline_type;
        initialize_generation_config("StableDiffusionXLPipeline");
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_clip_text_encoder_with_projection->reshape(batch_size_multiplier);

        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &DiffusionPipeline::derived_adapters);
        // updated_properies are for passing to the pipeline subcomponents only, not for the generation config
        m_clip_text_encoder->compile(text_encode_device, *updated_properties);
        m_clip_text_encoder_with_projection->compile(text_encode_device, *updated_properties);
        m_unet->compile(denoise_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<CLIPTextModel> clip_text_encoder = m_clip_text_encoder->clone();
        std::shared_ptr<CLIPTextModelWithProjection> clip_text_encoder_with_projection = std::static_pointer_cast<CLIPTextModelWithProjection>(m_clip_text_encoder_with_projection->clone());
        std::shared_ptr<UNet2DConditionModel> unet = std::make_shared<UNet2DConditionModel>(m_unet->clone());
        std::shared_ptr<StableDiffusionXLPipeline> pipeline = std::make_shared<StableDiffusionXLPipeline>(
            m_pipeline_type,
            *clip_text_encoder,
            *clip_text_encoder_with_projection,
            *unet,
            *vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG

        std::vector<float> time_ids = {static_cast<float>(generation_config.width),
                                       static_cast<float>(generation_config.height),
                                       0,
                                       0,
                                       static_cast<float>(generation_config.width),
                                       static_cast<float>(generation_config.height),
                                       };
        ov::Tensor add_time_ids(ov::element::f32, {batch_size_multiplier, time_ids.size()});
        float* add_time_ids_data = add_time_ids.data<float>();
        std::copy(time_ids.begin(), time_ids.end(), add_time_ids_data);

        if (batch_size_multiplier > 1) {
            std::copy(time_ids.begin(), time_ids.end(), add_time_ids_data + time_ids.size());
        }

        std::string prompt_2_str = generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;
        std::string negative_prompt_1_str = generation_config.negative_prompt != std::nullopt ? *generation_config.negative_prompt : std::string{};
        std::string negative_prompt_2_str = generation_config.negative_prompt_2 != std::nullopt ? *generation_config.negative_prompt_2 : negative_prompt_1_str;

        // see https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L423-L427
        bool force_zeros_for_empty_prompt = generation_config.negative_prompt == std::nullopt && m_force_zeros_for_empty_prompt;
        bool compute_negative_prompt = !force_zeros_for_empty_prompt && batch_size_multiplier > 1;

        size_t idx_hidden_state_1 = m_clip_text_encoder->get_config().num_hidden_layers + 1;
        size_t idx_hidden_state_2 = m_clip_text_encoder_with_projection->get_config().num_hidden_layers + 1;

        ov::Tensor encoder_hidden_states(ov::element::f32, {}), add_text_embeds(ov::element::f32, {});

        if (compute_negative_prompt) {
            auto infer_start = std::chrono::steady_clock::now();
            add_text_embeds = m_clip_text_encoder_with_projection->infer(positive_prompt, negative_prompt_1_str, batch_size_multiplier > 1);
            auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
            m_perf_metrics.encoder_inference_duration["text_encoder_2"] = infer_duration;
            infer_start = std::chrono::steady_clock::now();
            m_clip_text_encoder->infer(prompt_2_str, negative_prompt_2_str, batch_size_multiplier > 1);
            infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
            m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

            // prompt_embeds = prompt_embeds.hidden_states[-2]
            ov::Tensor encoder_hidden_states_1 = m_clip_text_encoder->get_output_tensor(idx_hidden_state_1);
            ov::Tensor encoder_hidden_states_2 = m_clip_text_encoder_with_projection->get_output_tensor(idx_hidden_state_2);

            encoder_hidden_states = numpy_utils::concat(encoder_hidden_states_1, encoder_hidden_states_2, -1);
        } else {
            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor add_text_embeds_positive = m_clip_text_encoder_with_projection->infer(positive_prompt, negative_prompt_1_str, false);
            auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
            m_perf_metrics.encoder_inference_duration["text_encoder_2"] = infer_duration;
            infer_start = std::chrono::steady_clock::now();
            m_clip_text_encoder->infer(prompt_2_str, negative_prompt_2_str, false);
            infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
            m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

            ov::Tensor encoder_hidden_states_1_positive = m_clip_text_encoder->get_output_tensor(idx_hidden_state_1);
            ov::Tensor encoder_hidden_states_2_positive = m_clip_text_encoder_with_projection->get_output_tensor(idx_hidden_state_2);

            ov::Shape ehs_1_shape = encoder_hidden_states_1_positive.get_shape();
            ov::Shape ehs_2_shape = encoder_hidden_states_2_positive.get_shape();

            OPENVINO_ASSERT(ehs_1_shape[0] == ehs_2_shape[0] && ehs_1_shape[1] == ehs_2_shape[1],
                            "Tensors for concatenation must have the same dimensions");

            ov::Shape add_text_embeds_shape = add_text_embeds_positive.get_shape();
            add_text_embeds_shape[0] *= batch_size_multiplier;
            ov::Shape encoder_hidden_states_shape = {ehs_1_shape[0] * batch_size_multiplier, ehs_1_shape[1], ehs_1_shape[2] + ehs_2_shape[2]};

            add_text_embeds.set_shape(add_text_embeds_shape);
            encoder_hidden_states.set_shape(encoder_hidden_states_shape);

            float * add_text_embeds_data = add_text_embeds.data<float>();
            float * encoder_hidden_states_data = encoder_hidden_states.data<float>();

            // apply force_zeros_for_empty_prompt
            if (batch_size_multiplier > 1) {
                size_t encoder_hidden_states_size = ov::shape_size(encoder_hidden_states_shape) / batch_size_multiplier;

                std::fill_n(add_text_embeds_data, add_text_embeds_positive.get_size(), 0.0f);
                std::fill_n(encoder_hidden_states_data, encoder_hidden_states_size, 0.0f);

                add_text_embeds_data += add_text_embeds_positive.get_size();
                encoder_hidden_states_data += encoder_hidden_states_size;
            }

            std::copy_n(add_text_embeds_positive.data<float>(), add_text_embeds_positive.get_size(), add_text_embeds_data);

            const float* ehs_1_data = encoder_hidden_states_1_positive.data<const float>();
            const float* ehs_2_data = encoder_hidden_states_2_positive.data<const float>();

            for (size_t i = 0; i < ehs_1_shape[0] * ehs_1_shape[1]; ++i,
                encoder_hidden_states_data += encoder_hidden_states_shape[2],
                ehs_1_data += ehs_1_shape[2], ehs_2_data += ehs_2_shape[2]) {
                std::memcpy(encoder_hidden_states_data                 , ehs_1_data, ehs_1_shape[2] * sizeof(float));
                std::memcpy(encoder_hidden_states_data + ehs_1_shape[2], ehs_2_data, ehs_2_shape[2] * sizeof(float));
            }
        }

        // replicate encoder hidden state to UNet model
        if (generation_config.num_images_per_prompt == 1) {
            // reuse output of text encoder directly w/o extra memory copy
            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);
            m_unet->set_hidden_states("text_embeds", add_text_embeds);
            m_unet->set_hidden_states("time_ids", add_time_ids);
        } else {
            ov::Shape enc_shape = encoder_hidden_states.get_shape();
            enc_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor encoder_hidden_states_repeated(encoder_hidden_states.get_element_type(), enc_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);

            ov::Shape t_emb_shape = add_text_embeds.get_shape();
            t_emb_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor add_text_embeds_repeated(add_text_embeds.get_element_type(), t_emb_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(add_text_embeds, add_text_embeds_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(add_text_embeds, add_text_embeds_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("text_embeds", add_text_embeds_repeated);

            ov::Shape t_ids_shape = add_time_ids.get_shape();
            t_ids_shape[0] *= generation_config.num_images_per_prompt;
            ov::Tensor add_time_ids_repeated(add_time_ids.get_element_type(), t_ids_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(add_time_ids, add_time_ids_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(add_time_ids, add_time_ids_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("time_ids", add_time_ids_repeated);
        }

        if (unet_config.time_cond_proj_dim >= 0) { // LCM
            ov::Tensor timestep_cond = get_guidance_scale_embedding(generation_config.guidance_scale - 1.0f, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", timestep_cond);
        }
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if (adapters) {
            if (auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_clip_text_encoder->set_adapters(adapters);
            m_clip_text_encoder_with_projection->set_adapters(adapters);
            m_unet->set_adapters(adapters);
        }
    }

    void export_model(const std::filesystem::path& export_path) override {
        m_unet->export_model(export_path / "unet");
        m_clip_text_encoder->export_model(export_path / "text_encoder");
        m_clip_text_encoder_with_projection->export_model(export_path / "text_encoder_2");
        m_vae->export_model(export_path);
    }

private:
    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_unet != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        // in case of image to image, the shape is computed based on initial image
        if (m_pipeline_type != PipelineType::IMAGE_2_IMAGE) {
            m_generation_config.height = unet_config.sample_size * vae_scale_factor;
            m_generation_config.width = unet_config.sample_size * vae_scale_factor;
        }

        if (class_name == "StableDiffusionXLPipeline" || class_name == "StableDiffusionXLImg2ImgPipeline" || class_name == "StableDiffusionXLInpaintPipeline") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_generation_config.guidance_scale = 5.0f;
                m_generation_config.num_inference_steps = 50;
                m_generation_config.strength = 1.0f;
            } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_generation_config.guidance_scale = 5.0f;
                m_generation_config.num_inference_steps = 50;
                m_generation_config.strength = 0.3f;
            } else if (m_pipeline_type == PipelineType::INPAINTING) {
                m_generation_config.guidance_scale = 7.5f;
                m_generation_config.num_inference_steps = 50;
                m_generation_config.strength = 0.9999f;
            }
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        const bool is_classifier_free_guidance = m_unet->do_classifier_free_guidance(generation_config.guidance_scale);
        const char * const pipeline_name = "Stable Diffusion XL";

        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt == std::nullopt, "Negative prompt is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ", pipeline_name);

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
        }
    }

    utils::SharedOptional<const ov::AnyMap> properties_for_text_encoder(const ov::AnyMap& properties, const std::string& tensor_name_prefix) {
        return update_adapters_in_properties(properties,
            [&tensor_name_prefix](const AdapterConfig& adapters) -> std::optional<AdapterConfig> {
                if(!adapters.get_tensor_name_prefix()) {
                    std::optional<AdapterConfig> updated_adapters = adapters;
                    updated_adapters->set_tensor_name_prefix(tensor_name_prefix);
                    return updated_adapters;
                }
                return std::nullopt;
        });
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    bool m_force_zeros_for_empty_prompt = true;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_with_projection = nullptr;
};

}  // namespace genai
}  // namespace ov
