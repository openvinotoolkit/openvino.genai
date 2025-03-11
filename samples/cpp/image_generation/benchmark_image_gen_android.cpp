// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/layout.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "openvino/genai/image_generation/image2image_pipeline.hpp"
#include "openvino/genai/image_generation/inpainting_pipeline.hpp"
#include <cxxopts.hpp>
#include <chrono>
#include <ctime>
#include "imwrite.hpp"
#include "load_image.hpp"
#include "progress_bar.hpp"
#include "../text_generation/misc.h"

static std::string file_tm_stamp;
static std::ofstream report_file;
#if defined(__ANDROID__)
    std::string output_path = "/data/local/tmp/";
#elif defined(__linux__)
    std::string output_path = "./";
#endif

inline float get_total_text_encoder_infer_duration(ov::genai::ImageGenerationPerfMetrics& metrics) {
    float text_encoder_duration = 0.0f;
    for(auto text_encoder : metrics.get_text_encoder_infer_duration()) {
        text_encoder_duration += text_encoder.second;
    }
    return text_encoder_duration;
}

int report_file_open(int argc, char* argv[], cxxopts::ParseResult& result) {

    std::string report_path = result["output_dir"].as<std::string>();

    file_tm_stamp = gettimestamp(gettimenow());

    report_path += std::string("report_stable_diffusion_android_NPU_") + file_tm_stamp + ".txt";
    report_file.open(report_path);
    if (!report_file) {
        std::cout << "Unable to open Log report file. Error code: " << errno << " "
                  << std::strerror(errno) << std::endl;
        return EXIT_FAILURE;
    }

    std::string report_str;
    report_str = "Running Stable diffusion CLI app " + std::string(argv[0])+ " with args ";
    for(int i=1; i<argc; i++)
        report_str+=" "+std::string(argv[i]);

    report_str+="\n";
    std::cout << report_str << std::endl;
    report_file << report_str;
    std::cout << "Report file path is: " << report_path << std::endl;

    return EXIT_SUCCESS;
}


inline void print_one_generate(ov::genai::ImageGenerationPerfMetrics& metrics, std::string prefix, int idx) {
    std::string report_str;

    std::string prefix_idx;
    prefix_idx = "[" + prefix + "-" + std::to_string(idx) + "]";
    report_str = "Total generate time: " + std::to_string(metrics.get_generate_duration())
              + " ms, total infer time:" + std::to_string(metrics.get_inference_duration())
              + " ms\n";
    report_str = report_str + prefix_idx + " text encoder infer time: " + std::to_string(get_total_text_encoder_infer_duration(metrics)) + " ms\n";
    report_file << report_str;

    float first_iter_time, other_iter_avg_time;
    float first_infer_time, other_infer_avg_time;
    metrics.get_first_and_other_iter_duration(first_iter_time, other_iter_avg_time);
    if (!metrics.raw_metrics.transformer_inference_durations.empty()) {
        metrics.get_first_and_other_trans_infer_duration(first_infer_time, other_infer_avg_time);
        report_str =  prefix_idx + " transformer iteration num:" + std::to_string(metrics.raw_metrics.iteration_durations.size())
                  + ", first iteration time:" + std::to_string(first_iter_time)
                  + " ms, other iteration avg time:" + std::to_string(other_iter_avg_time) + " ms\n";
        report_file << report_str;
        report_str = prefix_idx
                  + " transformer inference num:" + std::to_string(metrics.raw_metrics.transformer_inference_durations.size())
                  + ", first inference time:" + std::to_string(first_infer_time)
                  + " ms, other inference avg time:" + std::to_string(other_infer_avg_time) + " ms\n";
        report_file << report_str;
    } else {
        metrics.get_first_and_other_unet_infer_duration(first_infer_time, other_infer_avg_time);
        report_str =  prefix_idx + " unet iteration num:" + std::to_string(metrics.raw_metrics.iteration_durations.size())
                  +  ", first iteration time:" + std::to_string(first_iter_time)
                  + " ms, other iteration avg time:" + std::to_string(other_iter_avg_time) + " ms\n";
        report_file << report_str;
        report_str =  prefix_idx + " unet inference num:" + std::to_string(metrics.raw_metrics.unet_inference_durations.size())
                  + ", first inference time:" + std::to_string(first_infer_time)
                  + " ms, other inference avg time:" + std::to_string(other_infer_avg_time) + " ms\n";
        report_file << report_str;
    }
    report_str = prefix_idx + " vae encoder infer time:" + std::to_string(metrics.get_vae_encoder_infer_duration())
              + " ms, vae decoder infer time:" + std::to_string(metrics.get_vae_decoder_infer_duration()) + " ms\n";
    report_file << report_str;
    report_file.flush();
}

inline float calculate_average(std::vector<float>& durations) {
    float duration_mean = std::accumulate(durations.begin(),
                                           durations.end(),
                                           0.0f,
                                           [](const float& acc, const float& duration) -> float {
                                               return acc + duration;
                                           });
    if (!durations.empty()) {
        duration_mean /= durations.size();
    }
    return duration_mean;
}

inline void print_statistic(std::vector<ov::genai::ImageGenerationPerfMetrics>& warmup_metrics, std::vector<ov::genai::ImageGenerationPerfMetrics>& iter_metrics) {
    std::vector<float> generate_durations;
    std::vector<float> total_inference_durations;
    std::vector<float> text_encoder_durations;
    std::vector<float> vae_encoder_durations;
    std::vector<float> vae_decoder_durations;
    float load_time = 0.0f;
    int warmup_num = warmup_metrics.size();
    int iter_num = iter_metrics.size();

    float generate_warmup = 0.0f;
    float inference_warmup = 0.0f;
    if (!warmup_metrics.empty()) {
        generate_warmup = warmup_metrics[0].get_generate_duration();
        inference_warmup = warmup_metrics[0].get_inference_duration();
    }

    for (auto& metrics : iter_metrics) {
        generate_durations.emplace_back(metrics.get_generate_duration());
        total_inference_durations.emplace_back(metrics.get_inference_duration());
        vae_decoder_durations.emplace_back(metrics.get_vae_decoder_infer_duration());
        vae_encoder_durations.emplace_back(metrics.get_vae_encoder_infer_duration());
        text_encoder_durations.emplace_back(get_total_text_encoder_infer_duration(metrics));
        load_time = metrics.get_load_time();
    }

    float generate_mean = calculate_average(generate_durations);
    float inference_mean = calculate_average(total_inference_durations);
    float vae_decoder_mean = calculate_average(vae_decoder_durations);
    float vae_encoder_mean = calculate_average(vae_encoder_durations);
    float text_encoder_mean = calculate_average(text_encoder_durations);

    std::string report_str;
    report_str =  "\nTest finish, load time: " + std::to_string(load_time) +" ms\n";
    report_str +=  "Generate iteration number:" + std::to_string(iter_num) + ", for one iteration, generate avg time: " + std::to_string(generate_mean)
              + " ms, infer avg time:" + std::to_string(inference_mean)
              + " ms, all text encoders infer avg time:" + std::to_string(text_encoder_mean)
              + " ms, vae encoder infer avg time:" + std::to_string(vae_encoder_mean)
              + " ms, vae decoder infer avg time:" + std::to_string(vae_decoder_mean) + " ms\n";
    report_file << report_str;
    report_file.flush();
    std::cout << "Test completed. Check report file for results\n" << std::endl;
}

inline std::vector<std::string> device_string_to_triplet(const std::string& device_input) {
    std::vector<std::string> devices;
    std::istringstream stream(device_input);
    std::string device;

    // Split the device input string by commas
    while (std::getline(stream, device, ',')) {
        devices.push_back(device);
    }

    // Trim whitespace from each device name
    for (auto& dev : devices) {
        dev.erase(0, dev.find_first_not_of(" \t"));
        dev.erase(dev.find_last_not_of(" \t") + 1);
    }

    // Ensure exactly three devices
    if (devices.size() == 1) {
        return {devices[0], devices[0], devices[0]};
    } else if (devices.size() == 3) {
        return devices;
    } else {
        throw std::invalid_argument("The device specified by -d/--device must be a single device (e.g. -d \"GPU\"), "
                                    "or exactly 3 comma separated device names (e.g. -d \"CPU,NPU,GPU\")");
    }
}

void text2image(cxxopts::ParseResult& result) {
    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    auto devices = device_string_to_triplet(result["device"].as<std::string>());
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    const std::string output_dir = result["output_dir"].as<std::string>();
    bool npu_turbo = result["npu_turbo"].as<bool>();

#if defined(__ANDROID__)
    std::string npu_cache = "/data/local/tmp/.npu_cache";
#elif defined(__linux__)
    std::string npu_cache = "./.npu_cache";
#endif

    /*NOTE:Enable cache if blobs are required. Enabling cache increaes the run time*/
    ov::AnyMap properties = {ov::intel_npu::turbo(npu_turbo),
                            /*ov::cache_dir(npu_cache),*/
                            };

    ov::genai::Text2ImagePipeline pipe(models_path);
    if (result["reshape"].as<bool>()) {
        pipe.reshape(result["num_images_per_prompt"].as<size_t>(),
                     result["height"].as<size_t>(),
                     result["width"].as<size_t>(),
                     pipe.get_generation_config().guidance_scale);
    }
    std::cout << "Starting to compile... with devices " <<
                devices[0] << ":" << devices[1] << ":"<< devices[2] << std::endl;

    //TODO: Modify passing of properties based on npu_turbo flag inside the if{}else{}
    if (npu_turbo){
        std::cout << "<<<<<NPU TURBO ENABLED>>>>>"<< std::endl;
        pipe.compile(devices[0], devices[1], devices[2], properties);
    }
    else {
       pipe.compile(devices[0], devices[1], devices[2], properties);
    }

    std::cout << "Completed compile." << std::endl;
    std::cout << "Performing text2image" << std::endl;

    ov::genai::ImageGenerationConfig config = pipe.get_generation_config();

    config.width = result["width"].as<size_t>();
    config.height = result["height"].as<size_t>();
    config.negative_prompt = "ugly,deformed,bad quality";
    config.num_inference_steps = result["num_inference_steps"].as<size_t>();
    config.num_images_per_prompt = result["num_images_per_prompt"].as<size_t>();
    config.rng_seed = 2541156436;
    config.guidance_scale = 8.0;

    std::cout << "Following configs set for text2image: \n\tHeightXWidth = "
                << config.height << "X" << config.width <<
                "\n\tNegative Prompt: " << *config.negative_prompt <<
                "\n\tInference steps: " << config.num_inference_steps <<
                "\n\tImages per prompt: " << config.num_images_per_prompt <<
                "\n\tItertrations : "<<  num_iter <<
                "\n\tNPU Turbo : " << (npu_turbo ? "Enabled " : "Disabled ") <<
                std::endl;

    pipe.set_generation_config(config);

    std::cout << std::fixed << std::setprecision(2);
    std::vector<ov::genai::ImageGenerationPerfMetrics> warmup_metrics;
    for (size_t i = 0; i < num_warmup; i++) {
        pipe.generate(prompt);
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        warmup_metrics.emplace_back(metrics);
#ifdef DEBUG
        print_one_generate(metrics, "warmup", i);
#endif
    }

    std::vector<ov::genai::ImageGenerationPerfMetrics> iter_metrics;
    for (size_t i = 0; i < num_iter; i++) {
        ov::Tensor image = pipe.generate(prompt);
        ov::genai::ImageGenerationPerfMetrics metrics = pipe.get_performance_metrics();
        iter_metrics.emplace_back(metrics);
        std::string image_name = output_dir + "/" + file_tm_stamp +
                std::to_string(i) + ".bmp";
        std::cout << "Image stored in " << image_name << std::endl;
        imwrite(image_name, image, true);
        print_one_generate(metrics, "iter", i);
        std::cout << "Completed text2image iteration " << i << std::endl;
    }

    print_statistic(warmup_metrics, iter_metrics);
}


int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_image_generation", "Help command");

    options.add_options()
    //common parameters
    ("t,pipeline_type", "pipeline type: text2image", cxxopts::value<std::string>()->default_value("text2image"))
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("The Sky is blue because"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("NPU"))
    ("o,output_dir", "Path to save output image", cxxopts::value<std::string>()->default_value(output_path))
    ("is,num_inference_steps", "The number of inference steps used to denoise initial noised latent to final image", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("ni,num_images_per_prompt", "The number of images to generate per generate() call", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("i,image", "Image path", cxxopts::value<std::string>())
    ("x,npu_turbo", "Enable NPU Turbo", cxxopts::value<bool>()->default_value("false"))
    //special parameters of text2image pipeline
    ("w,width", "The width of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    ("ht,height", "The height of the resulting image", cxxopts::value<size_t>()->default_value(std::to_string(512)))
    ("r,reshape", "Reshape pipeline before compilation", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    if(report_file_open(argc, argv, result))
        return EXIT_FAILURE;

    std::string pipeline_type = result["pipeline_type"].as<std::string>();
    if (pipeline_type == "text2image") {
        text2image(result);
    } else {
        std::cout << "not support pipeline type: " << pipeline_type << std::endl;
    }

    report_file.close();
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
