// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_save_image.hpp"

#include "module_genai/module_factory.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <cstring>

namespace ov {
namespace genai {
namespace module {

GENAI_REGISTER_MODULE_SAME(SaveImageModule);

void SaveImageModule::print_static_config() {
    std::cout << R"(
  save_image:          # Module Name
    type: "SaveImageModule"
    description: "Save images to the output folder. Supported DataType: [OVTensor]"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "saved_image"
        type: "String"
      - name: "saved_images"
        type: "VecString"
    params:
      filename_prefix: "String"
    )" << std::endl;
}

namespace {
// Simple BMP file writer - no external dependencies required
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t file_type{0x4D42};  // "BM"
    uint32_t file_size{0};
    uint16_t reserved1{0};
    uint16_t reserved2{0};
    uint32_t offset_data{54};
};

struct BMPInfoHeader {
    uint32_t size{40};
    int32_t width{0};
    int32_t height{0};
    uint16_t planes{1};
    uint16_t bit_count{24};
    uint32_t compression{0};
    uint32_t size_image{0};
    int32_t x_pixels_per_meter{2835};
    int32_t y_pixels_per_meter{2835};
    uint32_t colors_used{0};
    uint32_t colors_important{0};
};
#pragma pack(pop)

bool write_bmp(const std::string& filepath, const uint8_t* data, int width, int height, int channels) {
    // BMP row size must be multiple of 4 bytes
    int row_stride = ((width * 3 + 3) / 4) * 4;

    BMPFileHeader file_header;
    BMPInfoHeader info_header;

    info_header.width = width;
    info_header.height = height;
    info_header.size_image = row_stride * height;
    file_header.file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + info_header.size_image;

    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));

    // BMP stores pixels bottom-to-top and in BGR order
    std::vector<uint8_t> row_buffer(row_stride, 0);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int src_idx = (y * width + x) * channels;
            int dst_idx = x * 3;

            if (channels == 1) {
                // Grayscale to BGR
                row_buffer[dst_idx + 0] = data[src_idx];
                row_buffer[dst_idx + 1] = data[src_idx];
                row_buffer[dst_idx + 2] = data[src_idx];
            } else if (channels == 3) {
                // RGB to BGR
                row_buffer[dst_idx + 0] = data[src_idx + 2];  // B
                row_buffer[dst_idx + 1] = data[src_idx + 1];  // G
                row_buffer[dst_idx + 2] = data[src_idx + 0];  // R
            } else if (channels == 4) {
                // RGBA to BGR (ignore alpha)
                row_buffer[dst_idx + 0] = data[src_idx + 2];  // B
                row_buffer[dst_idx + 1] = data[src_idx + 1];  // G
                row_buffer[dst_idx + 2] = data[src_idx + 0];  // R
            }
        }
        file.write(reinterpret_cast<const char*>(row_buffer.data()), row_stride);
    }

    return file.good();
}
}  // anonymous namespace


SaveImageModule::SaveImageModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    // Note: is_output_module should remain false since SaveImageModule is not a ResultModule.
    // It is a regular processing module that saves images as a side effect.
    if (!initialize()) {
        GENAI_ERR("Failed to initialize SaveImageModule");
    }
}

SaveImageModule::~SaveImageModule() {}

bool SaveImageModule::initialize() {
    const auto& params = module_desc->params;

    // Get filename_prefix parameter
    auto it_prefix = params.find("filename_prefix");
    if (it_prefix != params.end()) {
        m_filename_prefix = it_prefix->second;
    } else {
        m_filename_prefix = "output";
    }

    // Get output_folder parameter (optional, default to ./output)
    auto it_folder = params.find("output_folder");
    if (it_folder != params.end()) {
        m_output_folder = it_folder->second;
    } else {
        m_output_folder = "./output";
    }

    // Create output folder if it doesn't exist
    std::filesystem::path output_path(m_output_folder);
    if (!std::filesystem::exists(output_path)) {
        try {
            std::filesystem::create_directories(output_path);
        } catch (const std::exception& e) {
            GENAI_ERR("SaveImageModule[" + module_desc->name + "]: Failed to create output folder: " + e.what());
            return false;
        }
    }

    return true;
}

std::string SaveImageModule::generate_filename() {
    size_t seq_num = m_sequence_number.fetch_add(1);

    // Format: prefix_YYYYMMDD_HHMMSS_seqnum.bmp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_now);
#endif

    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", &tm_now);

    std::ostringstream filename;
    filename << m_filename_prefix << "_" << time_buffer << "_" << std::setfill('0') << std::setw(5) << seq_num << ".bmp";

    std::filesystem::path full_path = std::filesystem::path(m_output_folder) / filename.str();
    return full_path.string();
}

std::vector<std::string> SaveImageModule::save_tensor_as_image(const ov::Tensor& tensor, const std::string& filepath) {
    std::vector<std::string> saved_paths;
    auto shape = tensor.get_shape();
    auto element_type = tensor.get_element_type();

    // Expect tensor shape to be [batch, height, width, channels] or [height, width, channels]
    // or [batch, channels, height, width] (NCHW format)
    size_t batch = 1;
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;
    bool is_nhwc = true;

    if (shape.size() == 4) {
        batch = shape[0];
        if (shape[1] <= 4 && shape[2] > 4 && shape[3] > 4) {
            // NCHW format
            channels = shape[1];
            height = shape[2];
            width = shape[3];
            is_nhwc = false;
        } else {
            // NHWC format
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }
    } else if (shape.size() == 3) {
        if (shape[0] <= 4 && shape[1] > 4 && shape[2] > 4) {
            // CHW format
            channels = shape[0];
            height = shape[1];
            width = shape[2];
            is_nhwc = false;
        } else {
            // HWC format
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        }
    } else {
        GENAI_ERR("SaveImageModule: Unsupported tensor shape. Expected 3D or 4D tensor.");
        return saved_paths;  // Return empty vector
    }

    if (channels != 1 && channels != 3 && channels != 4) {
        GENAI_ERR("SaveImageModule: Unsupported number of channels: " + std::to_string(channels) + ". Expected 1, 3, or 4.");
        return saved_paths;  // Return empty vector
    }

    // For batch > 1, save each image separately
    size_t image_size = height * width * channels;
    std::vector<uint8_t> image_data(image_size);

    for (size_t b = 0; b < batch; b++) {
        std::string current_filepath = filepath;
        if (batch > 1) {
            // Insert batch index before extension
            size_t dot_pos = filepath.rfind('.');
            if (dot_pos != std::string::npos) {
                current_filepath = filepath.substr(0, dot_pos) + "_b" + std::to_string(b) + filepath.substr(dot_pos);
            } else {
                current_filepath = filepath + "_b" + std::to_string(b);
            }
        }

        // Convert tensor data to uint8
        if (element_type == ov::element::u8) {
            const uint8_t* data = tensor.data<uint8_t>() + b * image_size;
            if (is_nhwc) {
                std::memcpy(image_data.data(), data, image_size);
            } else {
                // Convert CHW to HWC
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        for (size_t c = 0; c < channels; c++) {
                            size_t chw_idx = c * height * width + h * width + w;
                            size_t hwc_idx = h * width * channels + w * channels + c;
                            image_data[hwc_idx] = data[chw_idx];
                        }
                    }
                }
            }
        } else if (element_type == ov::element::f32) {
            const float* data = tensor.data<float>() + b * image_size;
            if (is_nhwc) {
                for (size_t i = 0; i < image_size; i++) {
                    float val = data[i];
                    // Clamp to [0, 255] range
                    val = std::max(0.0f, std::min(255.0f, val * 255.0f));
                    image_data[i] = static_cast<uint8_t>(val);
                }
            } else {
                // Convert CHW to HWC
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        for (size_t c = 0; c < channels; c++) {
                            size_t chw_idx = c * height * width + h * width + w;
                            size_t hwc_idx = h * width * channels + w * channels + c;
                            float val = data[chw_idx];
                            val = std::max(0.0f, std::min(255.0f, val * 255.0f));
                            image_data[hwc_idx] = static_cast<uint8_t>(val);
                        }
                    }
                }
            }
        } else if (element_type == ov::element::f16) {
            const ov::float16* data = tensor.data<ov::float16>() + b * image_size;
            if (is_nhwc) {
                for (size_t i = 0; i < image_size; i++) {
                    float val = static_cast<float>(data[i]);
                    val = std::max(0.0f, std::min(255.0f, val * 255.0f));
                    image_data[i] = static_cast<uint8_t>(val);
                }
            } else {
                // Convert CHW to HWC
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        for (size_t c = 0; c < channels; c++) {
                            size_t chw_idx = c * height * width + h * width + w;
                            size_t hwc_idx = h * width * channels + w * channels + c;
                            float val = static_cast<float>(data[chw_idx]);
                            val = std::max(0.0f, std::min(255.0f, val * 255.0f));
                            image_data[hwc_idx] = static_cast<uint8_t>(val);
                        }
                    }
                }
            }
        } else {
            GENAI_ERR("SaveImageModule: Unsupported tensor element type: " + element_type.get_type_name());
            return saved_paths;  // Return empty vector on error
        }

        // Save as BMP using built-in writer
        bool result = write_bmp(current_filepath, image_data.data(),
                                static_cast<int>(width),
                                static_cast<int>(height),
                                static_cast<int>(channels));

        if (!result) {
            GENAI_ERR("SaveImageModule: Failed to save image to: " + current_filepath);
            return saved_paths;  // Return paths saved so far on error
        }

        saved_paths.push_back(current_filepath);
        GENAI_INFO("SaveImageModule: Saved image to: " + current_filepath);
    }

    return saved_paths;
}

void SaveImageModule::run() {
    GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    if (this->inputs.find("raw_data") == this->inputs.end()) {
        GENAI_ERR("SaveImageModule[" + module_desc->name + "]: 'raw_data' input not found");
        return;
    }

    auto& raw_data = this->inputs["raw_data"].data;
    std::vector<std::string> saved_filepaths;

    if (raw_data.is<ov::Tensor>()) {
        ov::Tensor tensor = raw_data.as<ov::Tensor>();
        std::string filepath = generate_filename();

        // Use save_tensor_as_image to get actual saved file paths (handles batch)
        auto paths = save_tensor_as_image(tensor, filepath);
        if (paths.empty()) {
            GENAI_ERR("SaveImageModule[" + module_desc->name + "]: Failed to save image");
        } else {
            saved_filepaths.insert(saved_filepaths.end(), paths.begin(), paths.end());
        }
    } else if (raw_data.is<std::vector<ov::Tensor>>()) {
        auto tensors = raw_data.as<std::vector<ov::Tensor>>();
        for (size_t i = 0; i < tensors.size(); i++) {
            std::string filepath = generate_filename();
            auto paths = save_tensor_as_image(tensors[i], filepath);
            if (paths.empty()) {
                GENAI_ERR("SaveImageModule[" + module_desc->name + "]: Failed to save image " + std::to_string(i));
            } else {
                saved_filepaths.insert(saved_filepaths.end(), paths.begin(), paths.end());
            }
        }
    } else {
        GENAI_ERR("SaveImageModule[" + module_desc->name + "]: Unsupported data type. Expected OVTensor or VecOVTensor.");
    }

    // Set output: single image path or vector of paths
    if (!saved_filepaths.empty()) {
        // Always set "saved_image" output with the first (or only) filepath
        this->outputs["saved_image"].data = saved_filepaths[0];
        // Set "saved_images" output with all filepaths
        this->outputs["saved_images"].data = saved_filepaths;

        GENAI_INFO("SaveImageModule[" + module_desc->name + "]: Output 'saved_image' = " + saved_filepaths[0]);
        if (saved_filepaths.size() > 1) {
            GENAI_INFO("SaveImageModule[" + module_desc->name + "]: Output 'saved_images' contains " + std::to_string(saved_filepaths.size()) + " files");
        }
    }
}

} // namespace module
} // namespace genai
} // namespace ov
