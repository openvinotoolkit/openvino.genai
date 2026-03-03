// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef ENABLE_OPENVINO_NEW_ARCH

#include "../utils/load_image.hpp"
#include "../utils/model_yaml.hpp"
#include "../utils/ut_modules_base.hpp"
#include "../utils/utils.hpp"

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

// Parameters for test:
//   string: mode ("text" or "vl")
//   string: device
using test_params = std::tuple<std::string, std::string>;
using namespace ov::genai::module;

// ============================================================================
// Test fixture
// ============================================================================

class LLMInferenceSDPAModuleTest : public ModuleTestBase,
                                    public ::testing::TestWithParam<test_params> {
private:
    std::string _mode;
    std::string _device;

    std::string _module_name = "sdpa_llm_infer";

public:
    static std::string get_test_case_name(const testing::TestParamInfo<test_params>& obj) {
        const auto& mode   = std::get<0>(obj.param);
        const auto& device = std::get<1>(obj.param);
        std::string result;
        result += "Device_" + device;
        result += "_Mode_" + mode;
        return result;
    }

    void SetUp() override {
        REGISTER_TEST_NAME();
        std::tie(_mode, _device) = GetParam();
    }

    void TearDown() override {}

protected:
    // ------------------------------------------------------------------
    // YAML generation — configure the module for text or VL mode
    // ------------------------------------------------------------------
    std::string get_yaml_content() override {
        YAML::Node config;
        config["global_context"]["model_type"] = "qwen3_5";

        YAML::Node pipeline_modules = config["pipeline_modules"];

        YAML::Node llm_sdpa;
        llm_sdpa["type"] = "LLMInferenceSDPAModule";
        llm_sdpa["device"] = _device;
        llm_sdpa["description"] = "LLM Inference SDPA Module test.";

        // ---- Inputs ----
        YAML::Node inputs;
        // input_ids is always required
        inputs.push_back(input_node("input_ids", "OVTensor"));

        if (_mode == "vl") {
            inputs.push_back(input_node("visual_embeds",   "OVTensor"));
            inputs.push_back(input_node("visual_pos_mask", "OVTensor"));
            inputs.push_back(input_node("grid_thw",        "OVTensor"));
        }
        llm_sdpa["inputs"] = inputs;

        // ---- Outputs ----
        YAML::Node outputs;
        outputs.push_back(output_node("generated_text", "String"));
        llm_sdpa["outputs"] = outputs;

        // ---- Params ----
        YAML::Node params;
        params["model_path"]     = TEST_MODEL::Qwen3_5_0_8B();
        params["max_new_tokens"] = "16";
        llm_sdpa["params"] = params;

        pipeline_modules[_module_name] = llm_sdpa;
        return YAML::Dump(config);
    }

    // ------------------------------------------------------------------
    // Input preparation
    // ------------------------------------------------------------------
    ov::AnyMap prepare_inputs() override {
        ov::AnyMap inputs;

        if (_mode == "text") {
            // Text mode: provide tokenized input_ids directly.
            // Token ids for a simple prompt (placeholder ids).
            std::vector<int64_t> token_values = {9707};  // e.g. "Hello"
            ov::Tensor input_ids(ov::element::i64, {1, token_values.size()});
            std::copy(token_values.begin(), token_values.end(), input_ids.data<int64_t>());
            inputs["input_ids"] = input_ids;

        } else {
            // VL mode: provide input_ids with vision placeholder tokens,
            // plus synthetic visual embeddings.
            //
            // Sequence layout: [vision_start] [img_tok × N_vis] [vision_end] [text_tok]
            //
            // Read model config to get hidden_size and special token ids dynamically
            auto model_cfg = ov::genai::modeling::models::Qwen3_5Config::from_json_file(
                TEST_MODEL::Qwen3_5_0_8B());
            const int64_t vision_start_id = model_cfg.vision_start_token_id;
            const int64_t image_token_id  = model_cfg.image_token_id;
            const int64_t vision_end_id   = model_cfg.vision_end_token_id;
            constexpr int64_t text_token_id = 9707;

            // grid_thw = [1, 4, 4] with spatial_merge_size=2 gives:
            //   n_vis = T * (H/merge) * (W/merge) = 1 * 2 * 2 = 4
            constexpr size_t n_vis   = 4;                   // visual tokens count
            constexpr size_t seq_len = 2 + n_vis + 1;      // start + vis×N + end + text
            const size_t hidden = static_cast<size_t>(model_cfg.text.hidden_size);

            // input_ids: [vision_start, img, img, img, img, vision_end, text]
            ov::Tensor input_ids(ov::element::i64, {1, seq_len});
            auto* ids = input_ids.data<int64_t>();
            ids[0] = vision_start_id;
            for (size_t i = 0; i < n_vis; ++i)
                ids[1 + i] = image_token_id;
            ids[1 + n_vis] = vision_end_id;
            ids[2 + n_vis] = text_token_id;
            inputs["input_ids"] = input_ids;

            // visual_embeds: compact visual embeddings [n_vis, hidden]
            // scatter_visual_embeds expects 2D [V, H] (flat across all images)
            ov::Tensor visual_embeds(ov::element::f32, {n_vis, hidden});
            std::fill_n(visual_embeds.data<float>(), n_vis * hidden, 0.01f);
            inputs["visual_embeds"] = visual_embeds;

            // visual_pos_mask: boolean [1, seq_len] — true at visual token positions
            ov::Tensor visual_pos_mask(ov::element::boolean, {1, seq_len});
            auto* mask = visual_pos_mask.data<bool>();
            mask[0] = false;                              // vision_start
            for (size_t i = 0; i < n_vis; ++i)
                mask[1 + i] = true;                       // image tokens
            mask[1 + n_vis] = false;                      // vision_end
            mask[2 + n_vis] = false;                      // text
            inputs["visual_pos_mask"] = visual_pos_mask;

            // grid_thw: [N_images, 3] — (T, H, W) for 3D MRoPE
            // Single image: T=1, H=4, W=4, spatial_merge_size=2 → n_vis = 1×2×2 = 4
            ov::Tensor grid_thw(ov::element::i64, {1, 3});
            auto* thw = grid_thw.data<int64_t>();
            thw[0] = 1;  // T
            thw[1] = 4;  // H
            thw[2] = 4;  // W
            inputs["grid_thw"] = grid_thw;
        }
        return inputs;
    }

    // ------------------------------------------------------------------
    // Output verification — check generated text against known output
    // ------------------------------------------------------------------
    void check_outputs(ov::genai::module::ModulePipeline& pipe) override {
        auto generated_text = pipe.get_output("generated_text").as<std::string>();

        if (std::getenv("VERBOSE"))
            std::cout << "[TEST:" << _mode << "] generated_text = [" << generated_text << "]" << std::endl;

        EXPECT_FALSE(generated_text.empty()) << "Generated text should not be empty";

        // Greedy decoding is deterministic — verify expected substrings.
        if (_mode == "text") {
            EXPECT_NE(generated_text.find(". 2020"), std::string::npos)
                << "Text-mode output should contain '. 2020', got: " << generated_text;
        } else {
            EXPECT_NE(generated_text.find("uality of the"), std::string::npos)
                << "VL-mode output should contain 'uality of the', got: " << generated_text;
        }
    }
};

// ============================================================================
// Test cases
// ============================================================================

TEST_P(LLMInferenceSDPAModuleTest, ModuleTest) {
    run();
}

// ============================================================================
// Parameterised instantiation
// ============================================================================

static std::vector<test_params> g_test_params = {
    // Text mode
    {"text", TEST_MODEL::get_device()},
    // VL mode
    {"vl",   TEST_MODEL::get_device()},
};

INSTANTIATE_TEST_SUITE_P(ModuleTestSuite,
                         LLMInferenceSDPAModuleTest,
                         ::testing::ValuesIn(g_test_params),
                         LLMInferenceSDPAModuleTest::get_test_case_name);

#endif  // ENABLE_OPENVINO_NEW_ARCH
