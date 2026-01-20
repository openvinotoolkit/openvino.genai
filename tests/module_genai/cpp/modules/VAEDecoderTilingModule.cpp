// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"

// Define test parameters:
// std::string: device;
// int: tile_size(sample_size);
using test_params = std::tuple<std::string, int>;
using namespace ov::genai::module;

class VAEDecoderTilingModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string _device;
    int _tile_size;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<0>(obj.param);
        int tile_size = std::get<1>(obj.param);
        std::string result;
        result += device;
        result += "_TileSize_" + std::to_string(tile_size);
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_device, _tile_size) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "zimage";
        YAML::Node pipeline_modules = config["pipeline_modules"];

        std::string vae_decoder_tiling_name = "vae_decoder_tiling";
        std::string vae_decoder_tiling_submodule_name = "vae_decoder_tiling_submodule";
        {
            YAML::Node cur_node;
            cur_node["type"] = "VAEDecoderTilingModule";
            cur_node["device"] = _device;
            cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["inputs"].push_back(input_node("latent", to_string(DataType::OVTensor)));
            cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
            cur_node["outputs"].push_back(output_node("image", to_string(DataType::OVTensor)));
            cur_node["params"] = YAML::Node();
            cur_node["params"]["tile_overlap_factor"] = "0.25";
            cur_node["params"]["sample_size"] = std::to_string(_tile_size);
            cur_node["params"]["model_path"] = TEST_MODEL::ZImage_Turbo_fp16_ov();
            cur_node["params"]["sub_module_name"] = vae_decoder_tiling_submodule_name;
            pipeline_modules[vae_decoder_tiling_name] = cur_node;
        }

        // Sub-module definition
        config["sub_modules"] = YAML::Node(YAML::NodeType::Sequence);
        {
            YAML::Node cur_submodule = YAML::Node(YAML::NodeType::Map);
            cur_submodule["name"] = vae_decoder_tiling_submodule_name;
            {
                YAML::Node cur_node;
                cur_node["type"] = "VAEDecoderModule";
                cur_node["device"] = _device;
                cur_node["inputs"] = YAML::Node(YAML::NodeType::Sequence);
                cur_node["inputs"].push_back(input_node("latents", to_string(DataType::OVTensor)));
                cur_node["outputs"] = YAML::Node(YAML::NodeType::Sequence);
                cur_node["outputs"].push_back(output_node("image", to_string(DataType::OVTensor)));
                cur_node["params"] = YAML::Node();
                cur_node["params"]["model_path"] = TEST_MODEL::ZImage_Turbo_fp16_ov();
                cur_node["params"]["enable_postprocess"] = "false";  // Tiling decoder, don't need to do post-process

                cur_submodule["vae_decoder"] = cur_node;
            }
            config["sub_modules"].push_back(cur_submodule);
        }

        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        auto latent = ut_randn_tensor(ov::Shape{1, 16, 15, 15}, 42);
        inputs["latent"] = latent;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto output = pipe.get_output("image").as<ov::Tensor>();
        EXPECT_EQ(output.get_element_type(), ov::element::u8) << "Expect output data type is u8";

        const std::vector<uint8_t> expected_output_1 = {186, 193, 169, 187, 189, 164, 182, 182};
        const std::vector<uint8_t> expected_output_2 = {190, 198, 173, 189, 192, 165, 183, 183};
        if (_tile_size == 64) {
            EXPECT_TRUE(compare_big_tensor<uint8_t>(output, expected_output_1, static_cast<uint8_t>(1)))
                << "latent do not match expected values";
        } else {
            EXPECT_TRUE(compare_big_tensor<uint8_t>(output, expected_output_2, static_cast<uint8_t>(1)))
                << "latent do not match expected values";
        }
    }
};

TEST_P(VAEDecoderTilingModuleTest, ModuleTest) {
    run();
}

static auto test_devices = std::vector<std::string>{TEST_MODEL::get_device()};
static auto test_tile_sizes = std::vector<int>{64, 1024};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         VAEDecoderTilingModuleTest,
                         ::testing::Combine(::testing::ValuesIn(test_devices), ::testing::ValuesIn(test_tile_sizes)),
                         VAEDecoderTilingModuleTest::get_test_case_name);