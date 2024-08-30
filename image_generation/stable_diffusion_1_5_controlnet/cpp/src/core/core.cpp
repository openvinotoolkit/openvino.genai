#include "core/core.hpp"

#include <algorithm>
#include <random>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/core.hpp"
#include "scheduler_lms_discrete.hpp"
#include "utils.hpp"

const size_t TOKENIZER_MODEL_MAX_LENGTH = 77;  // 'model_max_length' parameter from 'tokenizer_config.json'
const size_t VAE_SCALE_FACTOR = 8;

void exportTensorToTxt(const ov::Tensor& tensor, const std::string& filename) {
    if (tensor.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Tensor element type is not float32.");
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    const float* data = tensor.data<float>();
    size_t size = tensor.get_size();

    outfile << std::fixed << std::setprecision(8);

    for (size_t i = 0; i < size; ++i) {
        outfile << data[i] << "\n";
    }

    outfile.close();
    std::cout << "Tensor data has been exported to " << filename << std::endl;
}

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;

public:
    Timer(const std::string& scope) : m_start(std::chrono::steady_clock::now()) {
        (std::cout << scope << ": ").flush();
    }

    ~Timer() {
        auto m_end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(m_end - m_start).count() << " ms" << std::endl;
    }
};

ov::Tensor concat_twice(const ov::Tensor& input) {
    auto shape = input.get_shape();
    shape[0] = 2;
    ov::Tensor output_tensor(input.get_element_type(), shape);

    input.copy_to(ov::Tensor(output_tensor, {0, 0, 0, 0}, {1, shape[1], shape[2], shape[3]}));
    input.copy_to(ov::Tensor(output_tensor, {1, 0, 0, 0}, {2, shape[1], shape[2], shape[3]}));

    return output_tensor;
}

ov::Tensor load_controlnet_input(std::string controlnet_input) {
    ov::Tensor condition(ov::element::f32, {2, 3, 512, 512});
    std::ifstream controlnet_copy_file(controlnet_input.c_str(), std::ios::ate);
    OPENVINO_ASSERT(controlnet_copy_file.is_open(), "Cannot open ", controlnet_input.c_str());

    size_t file_size = controlnet_copy_file.tellg() / sizeof(float);
    OPENVINO_ASSERT(file_size >= condition.get_size(),
                    "Cannot generate ",
                    condition.get_shape(),
                    " with ",
                    controlnet_input.c_str(),
                    ". File size is small");

    controlnet_copy_file.seekg(0, std::ios::beg);
    for (size_t i = 0; i < condition.get_size(); ++i)
        controlnet_copy_file >> condition.data<float>()[i];

    return condition;
}

ov::Tensor randn_tensor(ov::Shape shape, std::string latent_path, uint32_t seed = 42) {
    ov::Tensor noise(ov::element::f32, shape);
    if (latent_path.empty()) {
        std::mt19937 gen{seed};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        std::generate_n(noise.data<float>(), noise.get_size(), [&]() {
            return normal(gen);
        });
    } else {
        // read from latent file
        std::ifstream latent_copy_file(latent_path.c_str(), std::ios::ate);
        OPENVINO_ASSERT(latent_copy_file.is_open(), "Cannot open ", latent_path.c_str());

        size_t file_size = latent_copy_file.tellg() / sizeof(float);
        OPENVINO_ASSERT(file_size >= noise.get_size(),
                        "Cannot generate ",
                        noise.get_shape(),
                        " with ",
                        latent_path.c_str(),
                        ". File size is small");

        latent_copy_file.seekg(0, std::ios::beg);
        for (size_t i = 0; i < noise.get_size(); ++i)
            latent_copy_file >> noise.data<float>()[i];
    }

    return noise;
}

void reshape_text_encoder(std::shared_ptr<ov::Model> model, size_t batch_size, size_t tokenizer_model_max_length) {
    ov::PartialShape input_shape = model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    input_shape[1] = tokenizer_model_max_length;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    model->reshape(idx_to_shape);
}

StableDiffusionControlnetPipeline::StableDiffusionControlnetPipeline(std::string model_path,
                                                                     std::string device,
                                                                     bool use_cache) {
    ov::Core core;

    if (use_cache)
        core.set_property(ov::cache_dir("./cache_dir"));

    core.add_extension(TOKENIZERS_LIBRARY_PATH);

    // Text encoder
    {
        Timer t("Loading and compiling text encoder");
        auto text_encoder_model = core.read_model(model_path + "/text_encoder.xml");
        reshape_text_encoder(text_encoder_model, 1, TOKENIZER_MODEL_MAX_LENGTH);
        text_encoder = core.compile_model(text_encoder_model, device);
    }

    // Unet
    {
        Timer t("Loading and compiling UNet");
        auto unet_model = core.read_model(model_path + "/unet_controlnet.xml");
        unet = core.compile_model(unet_model, device);
    }

    // Detector
    {
        Timer t("Loading and compiling Detector");
        detector.load(model_path + +"/openpose.xml");
    }

    // Controlnet
    {
        Timer t("Loading and compiling Controlnet");
        auto controlnet_model = core.read_model(model_path + "/controlnet-pose.xml");
        controlnet = core.compile_model(controlnet_model, device);
    }

    // VAE decoder
    {
        Timer t("Loading and compiling VAE decoder");
        auto vae_decoder_model = core.read_model(model_path + "/vae_decoder.xml");
        ov::preprocess::PrePostProcessor ppp(vae_decoder_model);
        ppp.output().model().set_layout("NCHW");
        ppp.output().tensor().set_layout("NHWC");
        vae_decoder = core.compile_model(vae_decoder_model = ppp.build(), device);
    }

    // Tokenizer
    {
        Timer t("Loading and compiling tokenizer");
        // Tokenizer model wil be loaded to CPU: OpenVINO Tokenizers can be inferred on a CPU device only.
        tokenizer = core.compile_model(model_path + "/tokenizer/openvino_tokenizer.xml", "CPU");
    }
}

ov::Tensor StableDiffusionControlnetPipeline::PreprocessEx(ov::Tensor pose,
                                                           int dst_height,
                                                           int dst_width,
                                                           StableDiffusionControlnetPipelinePreprocessMode mode,
                                                           int* pad_width,
                                                           int* pad_height) {
    auto shape = pose.get_shape();  // NHWC
    auto image_height = shape[1];
    auto image_width = shape[2];
    ov::Tensor resized_tensor = smart_resize(pose, dst_height, dst_width);

    switch (mode) {
    case JustResize:
        // resize
        break;
    case ResizeAndFill:
        // TODO: not supported yet
        break;
    case CropAndResize:
        // TODO: not supported yet
        break;
    default:
        break;
    }

    return Preprocess(resized_tensor, pad_width, pad_height);
}

ov::Tensor StableDiffusionControlnetPipeline::Preprocess(ov::Tensor pose, int* pad_width, int* pad_height) {
    auto shape = pose.get_shape();  // NHWC
    auto image_height = shape[1];
    auto image_width = shape[2];

    // resize
    float im_scale = std::min(512.f / image_height, 512.f / image_width);
    int result_width = static_cast<int>(im_scale * image_width);
    int result_height = static_cast<int>(im_scale * image_height);
    *pad_width = 512 - result_width;
    *pad_height = 512 - result_height;

    auto resized_tensor = lanczos_resize(pose, result_height, result_width);

    std::cout << "image_width: " << image_width << ", image_height" << image_height << std::endl;
    std::cout << "pad_width: " << *pad_width << ", pad_height" << *pad_height << std::endl;

    // pad the right bottom with 0
    auto padded_tensor = init_tensor_with_zeros({1, 512, 512, 3}, ov::element::u8);
    // Copy resized data to padded tensor
    auto* padded_data = padded_tensor.data<uint8_t>();
    auto* resized_data = resized_tensor.data<uint8_t>();
    for (int h = 0; h < result_height; ++h) {
        for (int w = 0; w < result_width; ++w) {
            for (int c = 0; c < 3; ++c) {
                padded_data[(h * 512 + w) * 3 + c] = resized_data[(h * result_width + w) * 3 + c];
            }
        }
    }

    // TODO: we may be interest with this
    // imwrite("controlnet_input_tensor.bmp", padded_tensor, true);

    // normalize to float32
    auto normalized_tensor = init_tensor_with_zeros({1, 512, 512, 3}, ov::element::f32);
    auto* normalized_data = normalized_tensor.data<float>();
    for (size_t i = 0; i < padded_tensor.get_byte_size(); ++i) {
        normalized_data[i] = static_cast<float>(padded_data[i]) / 255.f;
    }

    // transform to NCHW
    auto result_tensor = init_tensor_with_zeros({1, 3, 512, 512}, ov::element::f32);
    auto* output_data = result_tensor.data<float>();
    for (int h = 0; h < 512; ++h) {
        for (int w = 0; w < 512; ++w) {
            for (int c = 0; c < 3; ++c) {
                output_data[c * 512 * 512 + h * 512 + w] = normalized_data[h * 512 * 3 + w * 3 + c];
            }
        }
    }

    return result_tensor;
}

ov::Tensor StableDiffusionControlnetPipeline::Postprocess(const ov::Tensor& decoded_image,
                                                          int pad_height,
                                                          int pad_width,
                                                          int result_height,
                                                          int result_width) {
    auto shape = decoded_image.get_shape();  // NHWC
    size_t batch_size = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];
    size_t channels = shape[3];

    size_t unpad_height = height - pad_height;
    size_t unpad_width = width - pad_width;

    // unpadded tensor
    std::vector<size_t> new_shape = {batch_size, unpad_height, unpad_width, channels};
    ov::Tensor unpadded_image(decoded_image.get_element_type(), new_shape);

    const auto* input_data = decoded_image.data<float>();
    auto* output_data = unpadded_image.data<float>();

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t h = 0; h < unpad_height; ++h) {
            for (size_t w = 0; w < unpad_width; ++w) {
                for (size_t c = 0; c < channels; ++c) {
                    size_t input_index = n * height * width * channels + h * width * channels + w * channels + c;
                    size_t output_index =
                        n * unpad_height * unpad_width * channels + h * unpad_width * channels + w * channels + c;
                    output_data[output_index] = input_data[input_index];
                }
            }
        }
    }

    // resize to result size
    ov::Tensor result_image = smart_resize(unpadded_image, result_height, result_width);

    return result_image;
}

ov::Tensor StableDiffusionControlnetPipeline::TextEncoder(std::string& pos_prompt, std::string& neg_prompt) {
    const size_t HIDDEN_SIZE = static_cast<size_t>(text_encoder.output(0).get_partial_shape()[2].get_length());
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({1, TOKENIZER_MODEL_MAX_LENGTH});

    ov::InferRequest tokenizer_req = tokenizer.create_infer_request();
    ov::InferRequest text_encoder_req = text_encoder.create_infer_request();

    auto compute_text_embeddings = [&](std::string& prompt, ov::Tensor encoder_output_tensor) {
        ov::Tensor input_ids(ov::element::i64, input_ids_shape);
        std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), PAD_TOKEN_ID);

        // tokenization
        tokenizer_req.set_input_tensor(ov::Tensor{ov::element::string, {1}, &prompt});
        tokenizer_req.infer();
        ov::Tensor input_ids_token = tokenizer_req.get_tensor("input_ids");
        std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int64_t>());

        // text embeddings
        text_encoder_req.set_tensor("input_ids", input_ids);
        text_encoder_req.set_output_tensor(0, encoder_output_tensor);
        text_encoder_req.infer();
    };

    ov::Tensor text_embeddings(ov::element::f32, {2, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE});

    compute_text_embeddings(neg_prompt,
                            ov::Tensor(text_embeddings, {0, 0, 0}, {1, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE}));
    compute_text_embeddings(pos_prompt,
                            ov::Tensor(text_embeddings, {1, 0, 0}, {2, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE}));

    return text_embeddings;
}

ov::Tensor StableDiffusionControlnetPipeline::Unet(ov::InferRequest req,
                                                   ov::Tensor sample,
                                                   ov::Tensor timestep,
                                                   ov::Tensor text_embedding_1d,
                                                   float guidance_scale) {
    req.set_tensor("sample", sample);
    req.set_tensor("timestep", timestep);
    req.set_tensor("encoder_hidden_states", text_embedding_1d);

    req.infer();

    ov::Tensor noise_pred_tensor = req.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    // perform guidance
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] =
            noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
}

ov::Tensor StableDiffusionControlnetPipeline::ControlnetUnet(ov::InferRequest controlnet_infer_request,
                                                             ov::InferRequest unet_infer_request,
                                                             ov::Tensor sample,
                                                             ov::Tensor timestep,
                                                             ov::Tensor text_embedding_1d,
                                                             ov::Tensor controlnet_cond,
                                                             float guidance_scale,
                                                             float controlnet_conditioning_scale) {
    controlnet_infer_request.set_tensor("sample", sample);
    controlnet_infer_request.set_tensor("timestep", timestep);
    controlnet_infer_request.set_tensor("encoder_hidden_states", text_embedding_1d);
    controlnet_infer_request.set_tensor("controlnet_cond", controlnet_cond);

    controlnet_infer_request.infer();

    // setup unet request and params
    unet_infer_request.set_tensor("sample", sample);
    unet_infer_request.set_tensor("timestep", timestep);
    unet_infer_request.set_tensor("encoder_hidden_states", text_embedding_1d);
    size_t unet_input_idx = 3;

    for (size_t i = 0; i < controlnet.outputs().size(); i++, unet_input_idx++) {
        auto t = controlnet_infer_request.get_output_tensor(i);
        float* t_data = t.data<float>();
        for (size_t i = 0; i < t.get_size(); ++i) {
            t_data[i] = t_data[i] * controlnet_conditioning_scale;
        }
        unet_infer_request.set_input_tensor(unet_input_idx, t);
    }

    // inference
    unet_infer_request.infer();

    ov::Tensor noise_pred_tensor = unet_infer_request.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    // perform guidance
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] =
            noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
}

ov::Tensor StableDiffusionControlnetPipeline::VAE(ov::Tensor sample) {
    const float coeffs_const{1 / 0.18215};
    for (size_t i = 0; i < sample.get_size(); ++i)
        sample.data<float>()[i] *= coeffs_const;

    ov::InferRequest req = vae_decoder.create_infer_request();
    req.set_input_tensor(sample);
    req.infer();

    return req.get_output_tensor();
}

ov::Tensor StableDiffusionControlnetPipeline::Run(StableDiffusionControlnetPipelineParam& param) {
    Timer t("Running Stable Diffusion pipeline");

    ov::InferRequest unet_infer_request = unet.create_infer_request();
    ov::InferRequest controlnet_infer_request = controlnet.create_infer_request();

    ov::PartialShape sample_shape = unet.input("sample").get_partial_shape();
    ov::Tensor text_embeddings = TextEncoder(param.prompt, param.negative_prompt);

    ov::Tensor controlnet_input_tensor;
    int pad_width, pad_height = 0;
    int pose_width, pose_height = 0;
    bool has_input_image = param.input_image != "";

    if (has_input_image) {
        ov::Tensor input_image = read_image_to_tensor(param.input_image);
        std::vector<std::vector<float>> subset;
        std::vector<std::vector<float>> candidate;

        ov::Tensor pose_image = detector.forward(input_image, subset, candidate);
        pose_width = pose_image.get_shape()[2];
        pose_height = pose_image.get_shape()[1];
        ov::Tensor preprocessed_tensor;
        if (param.use_preprocess_ex) {
            preprocessed_tensor =
                PreprocessEx(pose_image, param.height, param.width, param.mode, &pad_width, &pad_height);
        } else {
            // default
            preprocessed_tensor = Preprocess(pose_image, &pad_width, &pad_height);
        }
        controlnet_input_tensor = concat_twice(preprocessed_tensor);
    }
    if (!param.controlnet_input.empty()) {
        controlnet_input_tensor = load_controlnet_input(param.controlnet_input);
    }

    std::shared_ptr<Scheduler> scheduler = std::make_shared<LMSDiscreteScheduler>();
    scheduler->set_timesteps(param.steps);
    std::vector<std::int64_t> timesteps = scheduler->get_timesteps();

    std::cout << "timesteps: " << std::endl;
    for (const auto ts : timesteps) {
        std::cout << ts << " ";
    }
    std::cout << std::endl;


    const size_t unet_in_channels = static_cast<size_t>(sample_shape[1].get_length());

    // latents are multiplied by 'init_noise_sigma'
    ov::Shape latent_shape = ov::Shape({1, unet_in_channels, 512 / VAE_SCALE_FACTOR, 512 / VAE_SCALE_FACTOR});
    ov::Shape latent_model_input_shape = latent_shape;
    ov::Tensor noise = randn_tensor(latent_shape, param.latent_path, param.seed);
    latent_model_input_shape[0] = 2;  // Unet accepts batch 2
    ov::Tensor latent(ov::element::f32, latent_shape), latent_model_input(ov::element::f32, latent_model_input_shape);
    for (size_t i = 0; i < noise.get_size(); ++i) {
        latent.data<float>()[i] = noise.data<float>()[i] * scheduler->get_init_noise_sigma();
    }
    for (size_t inference_step = 0; inference_step < param.steps; inference_step++) {
        std::cout << "running " << inference_step + 1 << "/" << param.steps << std::endl;

        // concat the same latent twice along a batch dimension
        latent.copy_to(
            ov::Tensor(latent_model_input, {0, 0, 0, 0}, {1, latent_shape[1], latent_shape[2], latent_shape[3]}));
        latent.copy_to(
            ov::Tensor(latent_model_input, {1, 0, 0, 0}, {2, latent_shape[1], latent_shape[2], latent_shape[3]}));

        scheduler->scale_model_input(latent_model_input, inference_step);

        ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
        ov::Tensor noisy_residual;
        if (has_input_image) {
            noisy_residual = ControlnetUnet(controlnet_infer_request,
                                            unet_infer_request,
                                            latent_model_input,
                                            timestep,
                                            text_embeddings,
                                            controlnet_input_tensor,
                                            param.guidance_scale,
                                            param.controlnet_conditioning_scale);
        } else {
            noisy_residual =
                Unet(unet_infer_request, latent_model_input, timestep, text_embeddings, param.guidance_scale);
        }

        latent = scheduler->step(noisy_residual, latent, inference_step)["latent"];
    }

    ov::Tensor decoded_image = VAE(latent);
    if (has_input_image) {
        if (param.use_preprocess_ex) {
            decoded_image = Postprocess(decoded_image, pad_height, pad_width, param.height, param.width);
        } else {
            // default
            decoded_image = Postprocess(decoded_image, pad_height, pad_width, pose_height, pose_width);
        }
    }
    return decoded_image;
}