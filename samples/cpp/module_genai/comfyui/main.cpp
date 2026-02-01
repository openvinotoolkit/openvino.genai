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

namespace fs = std::filesystem;

// ============================================================================
// Verbose Logging
// ============================================================================

enum class LogLevel { QUIET = 0, ERROR = 1, INFO = 2, DEBUG = 3 };

static LogLevel g_log_level = LogLevel::INFO;  // Default log level

#define LOG_ERROR(msg) do { if (g_log_level >= LogLevel::ERROR) { std::cerr << "[ERROR] " << msg << std::endl; } } while(0)
#define LOG_INFO(msg)  do { if (g_log_level >= LogLevel::INFO)  { std::cout << "[INFO] " << msg << std::endl; } } while(0)
#define LOG_DEBUG(msg) do { if (g_log_level >= LogLevel::DEBUG) { std::cout << "[DEBUG] " << msg << std::endl; } } while(0)
#define LOG_SUCCESS(msg) do { if (g_log_level >= LogLevel::INFO) { std::cout << "[SUCCESS] " << msg << std::endl; } } while(0)
#define LOG_WARNING(msg) do { if (g_log_level >= LogLevel::INFO) { std::cout << "[WARNING] " << msg << std::endl; } } while(0)

// ============================================================================
// Command Line Arguments
// ============================================================================

struct ProgramOptions {
    std::string json_file;
    std::string yaml_file;
    std::string model_path = "./ut_pipelines/Z-Image-Turbo-fp16-ov";
    std::string device = "CPU";
    std::string prompt;
    std::string output_file;
    int width = 0;
    int height = 0;
    int steps = 0;
    float guidance = 0.0f;
    int max_seq_len = 0;
    int tile_size = 0;
    int verbose = 2;  // 0=quiet, 1=error, 2=info (default), 3=debug
    bool show_help = false;
};

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n"
              << "OpenVINO GenAI ModulePipeline for ComfyUI\n"
              << "Runs image generation pipeline from ComfyUI both workflow  and API JSON config.\n\n"
              << "Options:\n"
              << "  --json <file>           ComfyUI JSON file (API or workflow format)\n"
              << "  --yaml <file>           YAML pipeline config file\n"
              << "  --model-path <path>     Model path base (default: ./models/)\n"
              << "  --device <device>       Device to run on (default: GPU.1)\n"
              << "  --prompt <text>         Text prompt for generation\n"
              << "  --output <file>         Output image filename (auto-generated if not specified)\n"
              << "  --width <int>           Image width (default: 1024)\n"
              << "  --height <int>          Image height (default: 1024)\n"
              << "  --steps <int>           Number of inference steps (default: 4)\n"
              << "  --guidance <float>      Guidance scale (default: 0.0)\n"
              << "  --max-seq-len <int>     Max sequence length (default: 512)\n"
              << "  --tile_size <int>       VAE decoder tile size (sample_size)\n"
              << "  --verbose <level>       Verbosity: 0=quiet, 1=error, 2=info (default), 3=debug\n"
              << "  --help                  Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --json workflow_api.json --prompt \"a cat\"\n"
              << "  " << program_name << " --yaml pipeline.yaml --output result.bmp\n"
              << std::endl;
}

// Forward declarations
bool parse_arguments(int argc, char* argv[], ProgramOptions& opts);
std::string read_file_content(const std::string& filepath);
bool load_and_validate_yaml(const std::string& yaml_file, std::string& yaml_content, ov::AnyMap& extracted_inputs);
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

    try {
        std::string yaml_content;
        ov::AnyMap extracted_inputs;  // Inputs extracted from ComfyUI JSON

        // ====================================================================
        // Step 1: Get YAML content (from JSON conversion or direct YAML file)
        // ====================================================================

        if (!opts.json_file.empty()) {
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
        } else if (!opts.yaml_file.empty()) {
            // Direct YAML input - load and validate
            if (!load_and_validate_yaml(opts.yaml_file, yaml_content, extracted_inputs)) {
                return EXIT_FAILURE;
            }
        }

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
        } else if (arg == "--yaml" && i + 1 < argc) {
            opts.yaml_file = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
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
        } else if (arg == "--steps" && i + 1 < argc) {
            opts.steps = std::stoi(argv[++i]);
        } else if (arg == "--guidance" && i + 1 < argc) {
            opts.guidance = std::stof(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            opts.max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--tile_size" && i + 1 < argc) {
            opts.tile_size = std::stoi(argv[++i]);
        } else if (arg == "--verbose" && i + 1 < argc) {
            opts.verbose = std::stoi(argv[++i]);
        } else {
            std::cerr << "[ERROR] Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    // Set global log level based on verbose option
    g_log_level = static_cast<LogLevel>(opts.verbose);

    if (!opts.show_help && opts.json_file.empty() && opts.yaml_file.empty()) {
        std::cerr << "[ERROR] Either --json or --yaml must be specified" << std::endl;
        return false;
    }

    return true;
}

// ============================================================================
// Read YAML file content
// ============================================================================

std::string read_file_content(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// ============================================================================
// Load and validate YAML config file
// ============================================================================

bool load_and_validate_yaml(const std::string& yaml_file,
                            std::string& yaml_content,
                            ov::AnyMap& extracted_inputs) {
    LOG_INFO("Loading YAML config from: " << yaml_file);

    if (!fs::exists(yaml_file)) {
        LOG_ERROR("YAML file not found: " << yaml_file);
        return false;
    }

    yaml_content = read_file_content(yaml_file);
    LOG_INFO("YAML config loaded successfully");

    // Validate YAML config
    LOG_INFO("Validating YAML config...");
    auto validation_result = ov::genai::module::ModulePipeline::validate_config_string(yaml_content);

    if (!validation_result.valid) {
        LOG_ERROR("YAML config validation failed:");
        for (const auto& error : validation_result.errors) {
            LOG_ERROR("  - " << error);
        }
        return false;
    }

    // Print warnings if any
    for (const auto& warning : validation_result.warnings) {
        LOG_WARNING(warning);
    }

    LOG_INFO("YAML config validation passed!");

    // Set default inputs for YAML mode
    extracted_inputs["width"] = 128;
    extracted_inputs["height"] = 128;
    extracted_inputs["num_inference_steps"] = 9;
    extracted_inputs["prompt"] = std::string("A chinese man with white T-shirt and blue jeans, "
        "standing in the forest, draw the light and shadow of the scene clearly, "
        "photo taken by Nikon D850, high resolution, detailed texture, draw full person.");
    extracted_inputs["negative_prompt"] = std::string("blurry ugly bad");

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
    if (g_log_level >= LogLevel::INFO) {
        std::cout << "[INFO] Final pipeline inputs:" << std::endl;
        std::cout << "  - prompt: \"" << inputs.at("prompt").as<std::string>() << "\"" << std::endl;
        std::cout << "  - negative_prompt: \"" << inputs.at("negative_prompt").as<std::string>() << "\"" << std::endl;
        std::cout << "  - width: " << inputs.at("width").as<int>() << std::endl;
        std::cout << "  - height: " << inputs.at("height").as<int>() << std::endl;
        std::cout << "  - batch_size: " << inputs.at("batch_size").as<int>() << std::endl;
        std::cout << "  - seed: " << inputs.at("seed").as<int64_t>() << std::endl;
        std::cout << "  - steps: " << inputs.at("num_inference_steps").as<int>() << std::endl;
        std::cout << "  - guidance: " << inputs.at("guidance_scale").as<float>() << std::endl;
        std::cout << "  - max_seq_len: " << inputs.at("max_sequence_length").as<int>() << std::endl;
        if (inputs.count("tile_size")) {
            std::cout << "  - tile_size: " << inputs.at("tile_size").as<int>() << std::endl;
        }
    }
}

void print_extracted_inputs(const ov::AnyMap& extracted_inputs) {
    if (g_log_level >= LogLevel::DEBUG) {
        std::cout << "[DEBUG] Extracted default inputs from ComfyUI JSON:" << std::endl;
        if (extracted_inputs.count("prompt")) {
            std::cout << "  - prompt: \"" << extracted_inputs.at("prompt").as<std::string>() << "\"" << std::endl;
        }
        if (extracted_inputs.count("negative_prompt")) {
            std::cout << "  - negative_prompt: \"" << extracted_inputs.at("negative_prompt").as<std::string>() << "\"" << std::endl;
        }
        if (extracted_inputs.count("width")) {
            std::cout << "  - width: " << extracted_inputs.at("width").as<int>() << std::endl;
        }
        if (extracted_inputs.count("height")) {
            std::cout << "  - height: " << extracted_inputs.at("height").as<int>() << std::endl;
        }
        if (extracted_inputs.count("batch_size")) {
            std::cout << "  - batch_size: " << extracted_inputs.at("batch_size").as<int>() << std::endl;
        }
        if (extracted_inputs.count("seed")) {
            std::cout << "  - seed: " << extracted_inputs.at("seed").as<int64_t>() << std::endl;
        }
        if (extracted_inputs.count("num_inference_steps")) {
            std::cout << "  - steps: " << extracted_inputs.at("num_inference_steps").as<int>() << std::endl;
        }
        if (extracted_inputs.count("guidance_scale")) {
            std::cout << "  - guidance: " << extracted_inputs.at("guidance_scale").as<float>() << std::endl;
        }
        if (extracted_inputs.count("tile_size")) {
            std::cout << "  - tile_size: " << extracted_inputs.at("tile_size").as<int>() << std::endl;
        }
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
    if (g_log_level >= LogLevel::DEBUG) {
        std::cout << "\n[DEBUG] ====== YAML Pipeline Config ======\n" << yaml_content
                  << "\n[DEBUG] ==================================\n" << std::endl;
    }
}

// ============================================================================
// Handle pipeline output - get and save/verify image
// ============================================================================

bool handle_pipeline_output(ov::genai::module::ModulePipeline& pipeline,
                            const std::string& output_file) {
    LOG_INFO("Getting output...");

    // Try to get the saved image path from the pipeline output
    auto saved_image_output = pipeline.get_output("saved_image");

    if (saved_image_output.is<std::string>()) {
        std::string saved_image_path = saved_image_output.as<std::string>();

        if (saved_image_path.empty()) {
            LOG_ERROR("saved_image output is empty");
            return false;
        }

        LOG_INFO("Saved image path: " << saved_image_path);

        // Verify the file exists
        if (fs::exists(saved_image_path)) {
            auto file_size = fs::file_size(saved_image_path);
            LOG_SUCCESS("Image saved successfully: " << saved_image_path
                      << " (" << file_size << " bytes)");
            return true;
        } else {
            LOG_ERROR("Saved image file does not exist: " << saved_image_path);
            return false;
        }
    }

    // Fallback: try to get raw image tensor (for backward compatibility)
    auto output = pipeline.get_output("image");

    if (output.is<ov::Tensor>()) {
        auto tensor = output.as<ov::Tensor>();
        auto shape = tensor.get_shape();

        if (g_log_level >= LogLevel::INFO) {
            std::cout << "[INFO] Output tensor shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "[INFO] Output tensor element type: " << tensor.get_element_type() << std::endl;
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

    LOG_ERROR("Output is neither a saved image path nor a tensor");
    return false;
}
