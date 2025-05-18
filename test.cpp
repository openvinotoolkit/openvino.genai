// 멀티 디바이스 (MULTI, AUTO, HETERO) 설정별 LLM 파이프라인 예제
// OpenVINO GenAI에서 다양한 디바이스 조합으로 실행 가능한 구조

#include "src/llm/pipeline_stateful.hpp"
#include "openvino/core/core.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov;

constexpr size_t MAX_PROMPT_LEN = 32;
constexpr size_t MIN_RESPONSE_LEN = 32;

AnyMap configure_properties_for_device(const std::string& device) {
    AnyMap properties;
    if (device.find("NPU") != std::string::npos) {
        properties["STATIC_PIPELINE"] = true;
        properties["MAX_PROMPT_LEN"] = MAX_PROMPT_LEN;
        properties["MIN_RESPONSE_LEN"] = MIN_RESPONSE_LEN;
        properties["NUM_STREAMS"] = 1;
        properties["CACHE_DIR"] = "./ov_cache";
        properties["PERFORMANCE_HINT"] = "LATENCY";
    } else if (device.find("MULTI") != std::string::npos) {
        properties["MULTI_DEVICE_PRIORITIES"] = "NPU,CPU";
        properties["CACHE_DIR"] = "./ov_cache";
    } else if (device.find("AUTO") != std::string::npos) {
        properties["CACHE_DIR"] = "./ov_cache";
    } else if (device.find("HETERO") != std::string::npos) {
        properties["CACHE_DIR"] = "./ov_cache";
    }
    return properties;
}

std::shared_ptr<Model> load_and_reshape_model(const std::filesystem::path& model_path, const std::string& device) {
    Core core;
    auto model = core.read_model(model_path.string());
    if (device.find("NPU") != std::string::npos || device.find("MULTI") != std::string::npos || device.find("AUTO") != std::string::npos) {
        model->reshape({
            {"input_ids", PartialShape{1, MAX_PROMPT_LEN}},
            {"attention_mask", PartialShape{1, MAX_PROMPT_LEN}}
        });
    }
    return model;
}

StatefulLLMPipeline build_pipeline(const std::filesystem::path& model_path,
                                   const Tokenizer& tokenizer,
                                   const std::string& device) {
    auto properties = configure_properties_for_device(device);
    auto model = load_and_reshape_model(model_path / "openvino_model.xml", device);

    GenerationConfig gen_config;
    gen_config.max_length = MAX_PROMPT_LEN + MIN_RESPONSE_LEN;
    gen_config.eos_token_id = tokenizer.get_eos_token_id();
    gen_config.num_return_sequences = 1;
    gen_config.do_sample = false;

    if (device.find("NPU") != std::string::npos ||
        device.find("MULTI") != std::string::npos ||
        device.find("AUTO") != std::string::npos) {
        gen_config.set_beam_search(false);
        gen_config.set_multinomial(false);
        gen_config.set_greedy(true);
    }

    return StatefulLLMPipeline(model, tokenizer, device, properties, gen_config);
}

int main() {
    std::string model_dir = "./Llama-2-7B-Chat-FP16";
    std::vector<std::string> device_list = {
        "MULTI:NPU,CPU",
        "AUTO",
        "HETERO:NPU,CPU"
    };

    Tokenizer tokenizer(model_dir);

    for (const auto& device : device_list) {
        std::cout << "\n[Running on device: " << device << "]\n" << std::endl;

        auto pipeline = build_pipeline(model_dir, tokenizer, device);
        std::string prompt = "What is the capital of France?";
        auto result = pipeline.generate(prompt);

        for (const auto& output : result.texts) {
            std::cout << "[Answer]: " << output << std::endl;
        }
    }

    return 0;
}

/*
📊 성능 비교 가이드 (예상 기준)
--------------------------------------------------
| 디바이스 모드     | 설명                         | 장점            | 단점              |
|------------------|------------------------------|-----------------|-------------------|
| MULTI:NPU,CPU    | NPU 우선, 실패 시 CPU 사용    | 안정성 + 성능 가능성 | 설정 필요          |
| AUTO             | 자동 선택 (사용 가능한 장치) | 설정 간편       | 선택 제어 어려움     |
| HETERO:NPU,CPU   | 노드 단위 디바이스 분할 실행 | 세밀한 제어 가능 | 복잡함 + 예외 가능성 |
--------------------------------------------------
*/
