// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"
#include "../utils/model_yaml.hpp"

using namespace ov::genai::module;

struct RandomLatentImageTestData {
    int width;
    int height;
    int batch_size;
    int num_images_per_prompt;
    int seed;
};

namespace TEST_DATA {

RandomLatentImageTestData latent_image_test_data() {
    RandomLatentImageTestData data {};
    data.width = 128;
    data.height = 128;
    data.batch_size = 1;
    data.num_images_per_prompt = 1;
    data.seed = 42;
    return data;
}

}

using test_params = std::tuple<RandomLatentImageTestData, std::string>;

class RandomLatentImageModuleTest : public ModuleTestBase, public ::testing::TestWithParam<test_params> {
private:
    std::string m_device;
    RandomLatentImageTestData m_test_data;
    float m_threshold = 1e-2;

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& device = std::get<1>(obj.param);
        return "device_" + device;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(m_test_data, m_device) = GetParam();
    }

    void TearDown() override {}

protected:
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "zimage";
        YAML::Node pipeline_modules = config["pipeline_modules"];
        YAML::Node latent_image;
        latent_image["type"] = "RandomLatentImageModule";
        latent_image["device"] = m_device;
        YAML::Node inputs;
        YAML::Node width;
        width["name"] = "width";
        width["type"] = "Int";
        inputs.push_back(width);
        YAML::Node height;
        height["name"] = "height";
        height["type"] = "Int";
        inputs.push_back(height);
        YAML::Node batch_size;
        batch_size["name"] = "batch_size";
        batch_size["type"] = "Int";
        inputs.push_back(batch_size);
        YAML::Node num_images_per_prompt;
        num_images_per_prompt["name"] = "num_images_per_prompt";
        num_images_per_prompt["type"] = "Int";
        inputs.push_back(num_images_per_prompt);
        YAML::Node seed;
        seed["name"] = "seed";
        seed["type"] = "Int";
        inputs.push_back(seed);
        latent_image["inputs"] = inputs;
        YAML::Node outputs;
        YAML::Node latents;
        latents["name"] = "latents";
        latents["type"] = "OVTensor";
        outputs.push_back(latents);
        latent_image["outputs"] = outputs;
        YAML::Node params;
        params["model_path"] = TEST_MODEL::ZImage_Turbo_fp16_ov();
        latent_image["params"] = params;
        pipeline_modules["latent_image"] = latent_image;
        return YAML::Dump(config);
    }

    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;
        inputs["width"] = m_test_data.width;
        inputs["height"] = m_test_data.height;
        inputs["batch_size"] = m_test_data.batch_size;
        inputs["num_images_per_prompt"] = m_test_data.num_images_per_prompt;
        inputs["seed"] = m_test_data.seed;
        return inputs;
    }

    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto latents = pipe.get_output("latents").as<ov::Tensor>();
        std::vector<float> expected_latents = {
            1.22192, -0.516964, 0.869636, 0.721333, 1.58856, 1.61822, -0.187125, -1.18831, -0.06342, -0.687744
        };
        EXPECT_TRUE(compare_shape(latents.get_shape(), ov::Shape {1, 16, 16, 16}))
            << "latents shape does not match expected shape";
        EXPECT_TRUE(compare_big_tensor(latents, expected_latents, m_threshold))
            << "latents do not match expected values within threshold " << m_threshold;
    }
};

TEST_P(RandomLatentImageModuleTest, ModuleTest) {
    run();
}

namespace latent_image_test {

auto test_data = std::vector<RandomLatentImageTestData> {TEST_DATA::latent_image_test_data()};
auto test_devices = std::vector<std::string> {TEST_MODEL::get_device()};

}

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         RandomLatentImageModuleTest,
                         ::testing::Combine(::testing::ValuesIn(latent_image_test::test_data),
                                            ::testing::ValuesIn(latent_image_test::test_devices)),
                         RandomLatentImageModuleTest::get_test_case_name);


