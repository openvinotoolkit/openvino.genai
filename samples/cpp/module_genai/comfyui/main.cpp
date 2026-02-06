// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file module_pipeline_comfyui.cpp
 * @brief Standalone tool to run OpenVINO GenAI ModulePipeline from ComfyUI JSON or YAML config
 *
 * Usage:
 *   module_pipeline_comfyui --json <comfyui.json> [options]
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <any>
#include <map>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// OpenVINO GenAI ModulePipeline
#include "openvino/genai/module_genai/pipeline.hpp"
#include "../utils/vision_utils.hpp"
#include "../utils/log_utils.hpp"

namespace fs = std::filesystem;

// ============================================================================
// Command Line Arguments
// ============================================================================

struct ProgramOptions {
    std::string json_file;  // Required parameter
    std::string model_path;  // Required parameter
    std::string device = "CPU";
    std::string prompt;
    std::string output_file;
    int width = 0;
    int height = 0;
    int num_frames = 0;
    int steps = 0;
    float guidance = 0.0f;
    int max_seq_len = 0;
    int tile_size = 0;
    int use_tiling = -1;  // -1=auto (default for wan2.1), 0=disable, 1=enable
    int verbose = 2;  // 0=quiet, 1=error, 2=info (default), 3=debug
    bool show_help = false;
};

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "OpenVINO GenAI ModulePipeline for ComfyUI\n"
              << "Runs image/video generation pipeline from ComfyUI workflow or API JSON config.\n\n"
              << "Options:\n"
              << "  --json <file>           ComfyUI JSON file (required, API or workflow format)\n"
              << "  --model_path <path>     Model path base (required)\n"
              << "  --device <device>       Device to run on (default: CPU)\n"
              << "  --prompt <text>         Text prompt for generation\n"
              << "  --output <file>         Output image/video filename (auto-generated if not specified)\n"
              << "  --width <int>           Image/video width (default: from JSON)\n"
              << "  --height <int>          Image/video height (default: from JSON)\n"
              << "  --num_frames <int>      Number of video frames (for video generation)\n"
              << "  --steps <int>           Number of inference steps (default: from JSON)\n"
              << "  --guidance <float>      Guidance scale (default: from JSON)\n"
              << "  --max_seq_len <int>     Max sequence length (default: 512)\n"
              << "  --tile_size <int>       VAE decoder tile size in pixels (default: 256)\n"
              << "  --use_tiling <0|1>      VAE tiling mode:\n"
              << "                            -1 = auto (default, enabled for Wan 2.1)\n"
              << "                             0 = disable tiling\n"
              << "                             1 = enable tiling\n"
              << "  --verbose <level>       Verbosity: 0=quiet, 1=error, 2=info (default), 3=debug\n"
              << "  --help                  Show this help message\n\n"
              << "Examples:\n"
              << "  # Image generation with Z-Image-Turbo\n"
              << "  " << program_name << " --json workflow.json --model_path ./models/Z-Image-Turbo\n\n"
              << "  # Video generation with Wan 2.1 on GPU (tiling auto-enabled)\n"
              << "  " << program_name << " --json wan2.1_t2v.json --model_path ./models/Wan2.1-T2V --device GPU\n\n"
              << "  # Video generation with tiling disabled\n"
              << "  " << program_name << " --json wan2.1_t2v.json --model_path ./models/Wan2.1-T2V --use_tiling 0\n"
              << std::endl;
}

// Forward declarations
bool parse_arguments(int argc, char* argv[], ProgramOptions& opts);
ov::AnyMap build_pipeline_inputs(const ov::AnyMap& extracted_inputs, const ProgramOptions& opts);
void print_pipeline_inputs(const ov::AnyMap& inputs);
void print_extracted_inputs(const ov::AnyMap& extracted_inputs);
void save_and_print_yaml_debug(const std::string& yaml_content, const std::string& base_filename);
bool handle_pipeline_output(ov::genai::module::ModulePipeline& pipeline, const std::string& output_file);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ProgramOptions opts;

    if (!parse_arguments(argc, argv, opts)) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    if (opts.show_help) {
        print_help(argv[0]);
        return EXIT_SUCCESS;
    }

    // Validate required parameters
    if (opts.json_file.empty()) {
        LOG_ERROR("--json is required. Please specify the ComfyUI JSON file.");
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    if (opts.model_path.empty()) {
        LOG_ERROR("--model_path is required. Please specify the path to the model directory.");
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        std::string yaml_content;
        ov::AnyMap extracted_inputs;  // Inputs extracted from ComfyUI JSON

        // ====================================================================
        // Step 1: Get YAML content from JSON conversion
        // ====================================================================

        // Convert ComfyUI JSON to YAML using ModulePipeline API
        LOG_INFO("Loading ComfyUI JSON from: " << opts.json_file);

        if (!fs::exists(opts.json_file)) {
            LOG_ERROR("JSON file not found: " << opts.json_file);
            return EXIT_FAILURE;
        }

        // Set conversion options via pipeline_inputs
        extracted_inputs["model_path"] = opts.model_path;
        extracted_inputs["device"] = opts.device;
        if (opts.tile_size != 0) {
            extracted_inputs["tile_size"] = opts.tile_size;
        }
        if (opts.use_tiling != -1) {
            extracted_inputs["use_tiling"] = (opts.use_tiling == 1);
        }

        // Use the new ModulePipeline API for conversion and input extraction
        yaml_content = ov::genai::module::ModulePipeline::comfyui_json_to_yaml(
            opts.json_file, extracted_inputs);

        if (yaml_content.empty()) {
            LOG_ERROR("Failed to convert JSON to YAML");
            return EXIT_FAILURE;
        }

        LOG_INFO("Converted to YAML successfully");
        print_extracted_inputs(extracted_inputs);

        // Save and print YAML for debugging
        save_and_print_yaml_debug(yaml_content, opts.json_file);

        // ====================================================================
        // Step 2: Create ModulePipeline
        // ====================================================================

        LOG_INFO("Creating ModulePipeline...");
        ov::genai::module::ModulePipeline pipeline(yaml_content);
        LOG_INFO("ModulePipeline created successfully!");

        // ====================================================================
        // Step 3: Prepare inputs and run
        // ====================================================================

        ov::AnyMap inputs = build_pipeline_inputs(extracted_inputs, opts);
        print_pipeline_inputs(inputs);

        LOG_INFO("Running pipeline.generate()...");
        auto start_time = std::chrono::high_resolution_clock::now();

        pipeline.generate(inputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOG_INFO("Generation completed in " << duration.count() << " ms");

        // ====================================================================
        // Step 4: Get output and verify saved image
        // ====================================================================

        if (!handle_pipeline_output(pipeline, opts.output_file)) {
            return EXIT_FAILURE;
        }

        LOG_SUCCESS("Pipeline execution completed!");
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        LOG_ERROR("Exception: " << e.what());
        return EXIT_FAILURE;
    }
}

// ============================================================================
// Helper Function Implementations
// ============================================================================

bool parse_arguments(int argc, char* argv[], ProgramOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
            return true;
        } else if (arg == "--json" && i + 1 < argc) {
            opts.json_file = argv[++i];
        } else if (arg == "--model_path" && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            opts.device = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            opts.prompt = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            opts.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            opts.height = std::stoi(argv[++i]);
        } else if (arg == "--num_frames" && i + 1 < argc) {
            opts.num_frames = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            opts.steps = std::stoi(argv[++i]);
        } else if (arg == "--guidance" && i + 1 < argc) {
            opts.guidance = std::stof(argv[++i]);
        } else if (arg == "--max_seq_len" && i + 1 < argc) {
            opts.max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--tile_size" && i + 1 < argc) {
            opts.tile_size = std::stoi(argv[++i]);
        } else if (arg == "--use_tiling" && i + 1 < argc) {
            opts.use_tiling = std::stoi(argv[++i]);
        } else if (arg == "--verbose" && i + 1 < argc) {
            opts.verbose = std::stoi(argv[++i]);
        } else {
            std::cerr << "[ERROR] Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    // Set global log level based on verbose option
    log_utils::set_log_level(opts.verbose);

    return true;
}

// ============================================================================
// Build pipeline inputs from extracted values and command line options
// ============================================================================

ov::AnyMap build_pipeline_inputs(const ov::AnyMap& extracted_inputs,
                                  const ProgramOptions& opts) {
    ov::AnyMap inputs;

    // Copy extracted inputs from ComfyUI JSON (if available)
    if (extracted_inputs.count("prompt")) {
        inputs["prompt"] = extracted_inputs.at("prompt").as<std::string>();
    }
    if (extracted_inputs.count("negative_prompt")) {
        inputs["negative_prompt"] = extracted_inputs.at("negative_prompt").as<std::string>();
    }
    if (extracted_inputs.count("guidance_scale")) {
        inputs["guidance_scale"] = extracted_inputs.at("guidance_scale").as<float>();
    }
    if (extracted_inputs.count("num_inference_steps")) {
        inputs["num_inference_steps"] = extracted_inputs.at("num_inference_steps").as<int>();
    }
    if (extracted_inputs.count("width")) {
        inputs["width"] = extracted_inputs.at("width").as<int>();
    }
    if (extracted_inputs.count("height")) {
        inputs["height"] = extracted_inputs.at("height").as<int>();
    }
    if (extracted_inputs.count("num_frames")) {
        inputs["num_frames"] = extracted_inputs.at("num_frames").as<int>();
    }
    if (extracted_inputs.count("max_sequence_length")) {
        inputs["max_sequence_length"] = extracted_inputs.at("max_sequence_length").as<int>();
    }
    if (extracted_inputs.count("batch_size")) {
        inputs["batch_size"] = extracted_inputs.at("batch_size").as<int>();
    }
    if (extracted_inputs.count("seed")) {
        inputs["seed"] = extracted_inputs.at("seed").as<int64_t>();
    }
    if (extracted_inputs.count("tile_size")) {
        inputs["tile_size"] = extracted_inputs.at("tile_size").as<int>();
    }

    // Command line arguments override extracted defaults
    if (!opts.prompt.empty()) {
        inputs["prompt"] = opts.prompt;
    }
    if (opts.width != 0) {
        inputs["width"] = opts.width;
    }
    if (opts.height != 0) {
        inputs["height"] = opts.height;
    }
    if (opts.num_frames != 0) {
        inputs["num_frames"] = opts.num_frames;
    }
    if (opts.steps != 0) {
        inputs["num_inference_steps"] = opts.steps;
    }
    if (opts.guidance != 0.0f) {
        inputs["guidance_scale"] = opts.guidance;
    }
    if (opts.max_seq_len != 0) {
        inputs["max_sequence_length"] = opts.max_seq_len;
    }
    if (opts.tile_size != 0) {
        inputs["tile_size"] = opts.tile_size;
    }

    // Ensure required inputs have default values
    if (!inputs.count("prompt") || inputs["prompt"].as<std::string>().empty()) {
        inputs["prompt"] = opts.prompt;
    }
    if (!inputs.count("width")) {
        inputs["width"] = opts.width;
    }
    if (!inputs.count("height")) {
        inputs["height"] = opts.height;
    }
    if (!inputs.count("num_inference_steps")) {
        inputs["num_inference_steps"] = opts.steps;
    }
    if (!inputs.count("guidance_scale")) {
        inputs["guidance_scale"] = opts.guidance;
    }
    if (!inputs.count("max_sequence_length")) {
        inputs["max_sequence_length"] = opts.max_seq_len;
    }
    if (!inputs.count("batch_size")) {
        inputs["batch_size"] = 1;
    }
    if (!inputs.count("seed")) {
        inputs["seed"] = static_cast<int64_t>(42);
    }
    if (!inputs.count("negative_prompt")) {
        inputs["negative_prompt"] = std::string("");
    }

    return inputs;
}

void print_pipeline_inputs(const ov::AnyMap& inputs) {
    LOG_INFO("Final pipeline inputs:");
    LOG_INFO("  - prompt: \"" << inputs.at("prompt").as<std::string>() << "\"");
    LOG_INFO("  - negative_prompt: \"" << inputs.at("negative_prompt").as<std::string>() << "\"");
    LOG_INFO("  - width: " << inputs.at("width").as<int>());
    LOG_INFO("  - height: " << inputs.at("height").as<int>());
    if (inputs.count("num_frames")) {
        LOG_INFO("  - num_frames: " << inputs.at("num_frames").as<int>());
    }
    LOG_INFO("  - batch_size: " << inputs.at("batch_size").as<int>());
    LOG_INFO("  - seed: " << inputs.at("seed").as<int64_t>());
    LOG_INFO("  - steps: " << inputs.at("num_inference_steps").as<int>());
    LOG_INFO("  - guidance: " << inputs.at("guidance_scale").as<float>());
    LOG_INFO("  - max_seq_len: " << inputs.at("max_sequence_length").as<int>());
    if (inputs.count("tile_size")) {
        LOG_INFO("  - tile_size: " << inputs.at("tile_size").as<int>());
    }
}

void print_extracted_inputs(const ov::AnyMap& extracted_inputs) {
    LOG_DEBUG("Extracted default inputs from ComfyUI JSON:");
    if (extracted_inputs.count("prompt")) {
        LOG_DEBUG("  - prompt: \"" << extracted_inputs.at("prompt").as<std::string>() << "\"");
    }
    if (extracted_inputs.count("negative_prompt")) {
        LOG_DEBUG("  - negative_prompt: \"" << extracted_inputs.at("negative_prompt").as<std::string>() << "\"");
    }
    if (extracted_inputs.count("width")) {
        LOG_DEBUG("  - width: " << extracted_inputs.at("width").as<int>());
    }
    if (extracted_inputs.count("height")) {
        LOG_DEBUG("  - height: " << extracted_inputs.at("height").as<int>());
    }
    if (extracted_inputs.count("num_frames")) {
        LOG_DEBUG("  - num_frames: " << extracted_inputs.at("num_frames").as<int>());
    }
    if (extracted_inputs.count("batch_size")) {
        LOG_DEBUG("  - batch_size: " << extracted_inputs.at("batch_size").as<int>());
    }
    if (extracted_inputs.count("seed")) {
        LOG_DEBUG("  - seed: " << extracted_inputs.at("seed").as<int64_t>());
    }
    if (extracted_inputs.count("num_inference_steps")) {
        LOG_DEBUG("  - steps: " << extracted_inputs.at("num_inference_steps").as<int>());
    }
    if (extracted_inputs.count("guidance_scale")) {
        LOG_DEBUG("  - guidance: " << extracted_inputs.at("guidance_scale").as<float>());
    }
    if (extracted_inputs.count("tile_size")) {
        LOG_DEBUG("  - tile_size: " << extracted_inputs.at("tile_size").as<int>());
    }
}

// ============================================================================
// Save and print YAML for debugging
// ============================================================================

void save_and_print_yaml_debug(const std::string& yaml_content, const std::string& base_filename) {
    // Save generated YAML to file
    std::string yaml_debug_file = base_filename + ".generated.yaml";
    std::ofstream yaml_out(yaml_debug_file);
    if (yaml_out.is_open()) {
        yaml_out << yaml_content;
        yaml_out.close();
        LOG_INFO("Generated YAML saved to: " << yaml_debug_file);
    }

    // Print YAML content for debugging
    LOG_DEBUG("====== YAML Pipeline Config ======\n" << yaml_content << "\n======================================");
}

// ============================================================================
// Handle pipeline output - get and save/verify image or video
// ============================================================================

bool handle_pipeline_output(ov::genai::module::ModulePipeline& pipeline,
                            const std::string& output_file) {
    LOG_INFO("Getting output...");

    // Priority: saved_video > saved_image > image tensor

    // 1. Try to get saved_video path (from SaveVideoModule)
    try {
        auto saved_video_output = pipeline.get_output("saved_video");
        if (saved_video_output.is<std::string>()) {
            std::string saved_video_path = saved_video_output.as<std::string>();
            if (!saved_video_path.empty()) {
                LOG_INFO("Video saved by pipeline to: " << saved_video_path);
                if (fs::exists(saved_video_path)) {
                    auto file_size = fs::file_size(saved_video_path);
                    LOG_SUCCESS("Video saved successfully: " << saved_video_path
                              << " (" << file_size << " bytes)");
                    return true;
                } else {
                    LOG_WARNING("Saved video file does not exist: " << saved_video_path);
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_DEBUG("saved_video output not available: " << e.what());
    }

    // 2. Try to get saved_image path (from SaveImageModule)
    try {
        auto saved_image_output = pipeline.get_output("saved_image");
        if (saved_image_output.is<std::string>()) {
            std::string saved_image_path = saved_image_output.as<std::string>();
            if (!saved_image_path.empty()) {
                LOG_INFO("Image saved by pipeline to: " << saved_image_path);
                if (fs::exists(saved_image_path)) {
                    auto file_size = fs::file_size(saved_image_path);
                    LOG_SUCCESS("Image saved successfully: " << saved_image_path
                              << " (" << file_size << " bytes)");
                    return true;
                } else {
                    LOG_WARNING("Saved image file does not exist: " << saved_image_path);
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_DEBUG("saved_image output not available: " << e.what());
    }

    // 3. Fallback: try to get raw image tensor
    try {
        auto output = pipeline.get_output("image");
        if (output.is<ov::Tensor>()) {
            auto tensor = output.as<ov::Tensor>();
            auto shape = tensor.get_shape();

            if (log_utils::get_log_level() >= log_utils::LogLevel::INFO) {
                std::ostringstream shape_ss;
                shape_ss << "[";
                for (size_t i = 0; i < shape.size(); ++i) {
                    shape_ss << shape[i];
                    if (i < shape.size() - 1) shape_ss << ", ";
                }
                shape_ss << "]";
                LOG_INFO("Output tensor shape: " << shape_ss.str());
                LOG_INFO("Output tensor element type: " << tensor.get_element_type());
            }

            // Generate output filename if not specified
            std::string output_filename = output_file;
            if (output_filename.empty()) {
                output_filename = image_utils::generate_output_filename("output", ".bmp");
            }

            LOG_INFO("Saving output image to: " << output_filename);

            if (image_utils::save_image_bmp(output_filename, tensor)) {
                LOG_SUCCESS("Image saved successfully: " << output_filename);
                return true;
            } else {
                LOG_ERROR("Failed to save image");
                return false;
            }
        }
    } catch (const std::exception& e) {
        LOG_DEBUG("image output not available: " << e.what());
    }

    LOG_ERROR("No valid output found (tried: saved_video, saved_image, image)");
    return false;
}
