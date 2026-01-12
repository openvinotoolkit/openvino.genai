// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../ut_modules_base.hpp"
#include <filesystem>

class SaveImageModuleBasic : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(SaveImageModuleBasic)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "image_data"
        type: "OVTensor"

  save_image:
    type: "SaveImageModule"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "pipeline_params.image_data"
    outputs:
      - name: "saved_image"
        type: "String"
      - name: "saved_images"
        type: "VecString"
    params:
      filename_prefix: "test_output"
      output_folder: "./output"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "saved_image"
        type: "String"
        source: "save_image.saved_image"
      - name: "saved_images"
        type: "VecString"
        source: "save_image.saved_images"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        // Create a test image tensor with shape [1, height, width, channels] (NHWC format)
        // Simple 64x64 RGB image
        ov::Tensor image_tensor(ov::element::u8, ov::Shape{1, 64, 64, 3});
        uint8_t* data = image_tensor.data<uint8_t>();

        // Fill with a gradient pattern for visual verification
        for (size_t h = 0; h < 64; h++) {
            for (size_t w = 0; w < 64; w++) {
                size_t idx = (h * 64 + w) * 3;
                data[idx + 0] = static_cast<uint8_t>(h * 4);       // R: vertical gradient
                data[idx + 1] = static_cast<uint8_t>(w * 4);       // G: horizontal gradient
                data[idx + 2] = static_cast<uint8_t>(128);         // B: constant
            }
        }

        inputs["image_data"] = image_tensor;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_image output from pipeline
        auto image_path = pipe.get_output("saved_image").as<std::string>();
        CHECK(!image_path.empty(), "saved_image output should not be empty");
        CHECK(std::filesystem::exists(image_path), "Output image file should exist: " + image_path);
        CHECK(image_path.find("test_output") != std::string::npos, "image path should contain 'test_output'");
        CHECK(image_path.find(".bmp") != std::string::npos, "image path should have .bmp extension");

        // Verify saved_images output (vector of paths)
        auto images = pipe.get_output("saved_images").as<std::vector<std::string>>();
        CHECK(images.size() == 1, "saved_images should contain 1 file path");
        CHECK(images[0] == image_path, "saved_images[0] should match saved_image output");

        // Clean up test file
        std::filesystem::remove(image_path);

        // Clean up output folder if empty
        std::filesystem::path output_folder("./output");
        if (std::filesystem::exists(output_folder) && std::filesystem::is_empty(output_folder)) {
            std::filesystem::remove(output_folder);
        }
    }
};

class SaveImageModuleFloat : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(SaveImageModuleFloat)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "image_data"
        type: "OVTensor"

  save_image:
    type: "SaveImageModule"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "pipeline_params.image_data"
    outputs:
      - name: "saved_image"
        type: "String"
      - name: "saved_images"
        type: "VecString"
    params:
      filename_prefix: "test_float"
      output_folder: "./output"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "saved_image"
        type: "String"
        source: "save_image.saved_image"
      - name: "saved_images"
        type: "VecString"
        source: "save_image.saved_images"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        // Create a test image tensor with float32 data (normalized 0-1 range)
        // Shape: [1, height, width, channels] (NHWC format)
        ov::Tensor image_tensor(ov::element::f32, ov::Shape{1, 32, 32, 3});
        float* data = image_tensor.data<float>();

        // Fill with a gradient pattern
        for (size_t h = 0; h < 32; h++) {
            for (size_t w = 0; w < 32; w++) {
                size_t idx = (h * 32 + w) * 3;
                data[idx + 0] = static_cast<float>(h) / 32.0f;   // R: vertical gradient
                data[idx + 1] = static_cast<float>(w) / 32.0f;   // G: horizontal gradient
                data[idx + 2] = 0.5f;                             // B: constant
            }
        }

        inputs["image_data"] = image_tensor;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_image output from pipeline
        auto image_path = pipe.get_output("saved_image").as<std::string>();
        CHECK(!image_path.empty(), "saved_image output should not be empty");
        CHECK(std::filesystem::exists(image_path), "Output image file should exist: " + image_path);
        CHECK(image_path.find("test_float") != std::string::npos, "image path should contain 'test_float'");

        // Clean up test file
        std::filesystem::remove(image_path);

        std::filesystem::path output_folder("./output");
        if (std::filesystem::exists(output_folder) && std::filesystem::is_empty(output_folder)) {
            std::filesystem::remove(output_folder);
        }
    }
};

class SaveImageModuleNCHW : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(SaveImageModuleNCHW)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "image_data"
        type: "OVTensor"

  save_image:
    type: "SaveImageModule"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "pipeline_params.image_data"
    outputs:
      - name: "saved_image"
        type: "String"
      - name: "saved_images"
        type: "VecString"
    params:
      filename_prefix: "test_nchw"
      output_folder: "./output"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "saved_image"
        type: "String"
        source: "save_image.saved_image"
      - name: "saved_images"
        type: "VecString"
        source: "save_image.saved_images"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        // Create a test image tensor in NCHW format (batch, channels, height, width)
        ov::Tensor image_tensor(ov::element::u8, ov::Shape{1, 3, 48, 48});
        uint8_t* data = image_tensor.data<uint8_t>();

        // Fill with a gradient pattern in NCHW format
        size_t height = 48;
        size_t width = 48;
        for (size_t c = 0; c < 3; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    size_t idx = c * height * width + h * width + w;
                    if (c == 0) {
                        data[idx] = static_cast<uint8_t>(h * 5);   // R channel
                    } else if (c == 1) {
                        data[idx] = static_cast<uint8_t>(w * 5);   // G channel
                    } else {
                        data[idx] = 200;                           // B channel
                    }
                }
            }
        }

        inputs["image_data"] = image_tensor;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_image output from pipeline
        auto image_path = pipe.get_output("saved_image").as<std::string>();
        CHECK(!image_path.empty(), "saved_image output should not be empty");
        CHECK(std::filesystem::exists(image_path), "Output image file should exist: " + image_path);
        CHECK(image_path.find("test_nchw") != std::string::npos, "image path should contain 'test_nchw'");

        // Clean up test file
        std::filesystem::remove(image_path);

        std::filesystem::path output_folder("./output");
        if (std::filesystem::exists(output_folder) && std::filesystem::is_empty(output_folder)) {
            std::filesystem::remove(output_folder);
        }
    }
};

class SaveImageModuleBatch : public ModuleTestBase {
public:
    DEFINE_MODULE_TEST_CONSTRUCTOR(SaveImageModuleBatch)

protected:
    std::string get_yaml_content() override {
        return R"(
global_context:
  model_type: "zimage"
pipeline_modules:

  pipeline_params:
    type: "ParameterModule"
    outputs:
      - name: "image_data"
        type: "OVTensor"

  save_image:
    type: "SaveImageModule"
    device: "CPU"
    inputs:
      - name: "raw_data"
        type: "OVTensor"
        source: "pipeline_params.image_data"
    outputs:
      - name: "saved_image"
        type: "String"
      - name: "saved_images"
        type: "VecString"
    params:
      filename_prefix: "test_batch"
      output_folder: "./output"

  pipeline_results:
    type: "ResultModule"
    device: "CPU"
    inputs:
      - name: "saved_image"
        type: "String"
        source: "save_image.saved_image"
      - name: "saved_images"
        type: "VecString"
        source: "save_image.saved_images"
)";
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        // Create a batch of 3 images in NHWC format
        ov::Tensor image_tensor(ov::element::u8, ov::Shape{3, 24, 24, 3});
        uint8_t* data = image_tensor.data<uint8_t>();

        size_t image_size = 24 * 24 * 3;
        for (size_t b = 0; b < 3; b++) {
            for (size_t h = 0; h < 24; h++) {
                for (size_t w = 0; w < 24; w++) {
                    size_t idx = b * image_size + (h * 24 + w) * 3;
                    // Different color for each batch image
                    data[idx + 0] = static_cast<uint8_t>(b == 0 ? 255 : h * 10);
                    data[idx + 1] = static_cast<uint8_t>(b == 1 ? 255 : w * 10);
                    data[idx + 2] = static_cast<uint8_t>(b == 2 ? 255 : 128);
                }
            }
        }

        inputs["image_data"] = image_tensor;
        return inputs;
    }

    void verify_outputs(ov::genai::module::ModulePipeline& pipe) override {
        // Verify saved_image output (first batch image path)
        auto image_path = pipe.get_output("saved_image").as<std::string>();
        CHECK(!image_path.empty(), "saved_image output should not be empty");
        CHECK(image_path.find("test_batch") != std::string::npos, "image path should contain 'test_batch'");

        // Verify saved_images output (all 3 batch image paths)
        auto images = pipe.get_output("saved_images").as<std::vector<std::string>>();
        CHECK(images.size() == 3, "saved_images should contain 3 file paths for batch of 3");

        // Verify all files exist and clean up
        for (const auto& filepath : images) {
            CHECK(std::filesystem::exists(filepath), "Output image file should exist: " + filepath);
            CHECK(filepath.find("test_batch") != std::string::npos, "image path should contain 'test_batch'");
            std::filesystem::remove(filepath);
        }

        std::filesystem::path output_folder("./output");
        if (std::filesystem::exists(output_folder) && std::filesystem::is_empty(output_folder)) {
            std::filesystem::remove(output_folder);
        }
    }
};

REGISTER_MODULE_TEST(SaveImageModuleBasic)
REGISTER_MODULE_TEST(SaveImageModuleFloat)
REGISTER_MODULE_TEST(SaveImageModuleNCHW)
REGISTER_MODULE_TEST(SaveImageModuleBatch)
