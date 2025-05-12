/ NPU 최적화를 위한 C++ 전체 파이프라인 개선 예제
// 적용 대상: StatefulLLMPipeline 사용 + OpenVINO GenAI 기반 추론

#include "src/cpp/src/llm/pipeline_stateful.hpp"

using namespace ov;

// 전역 상수 설정 (성능 최적화를 위한 고정 길이)
constexpr size_t MAX_PROMPT_LEN = 32;
constexpr size_t MIN_RESPONSE_LEN = 32;

AnyMap configure_npu_properties(const std::string& device) {
    AnyMap properties;
    if (device.find("NPU") != std::string::npos) {
        properties["STATIC_PIPELINE"] = true;
        properties["MAX_PROMPT_LEN"] = MAX_PROMPT_LEN;
        properties["MIN_RESPONSE_LEN"] = MIN_RESPONSE_LEN;
        properties["NUM_STREAMS"] = 1;
        properties["CACHE_DIR"] = "./ov_cache";
    }
    return properties;
}

std::shared_ptr<Model> load_model_for_npu(const std::filesystem::path& model_path, AnyMap& properties) {
    Core core;
    auto model = core.read_model(model_path.string());

    // 정적 shape 고정
    model->reshape({
        {"input_ids", PartialShape{1, MAX_PROMPT_LEN}},
        {"attention_mask", PartialShape{1, MAX_PROMPT_LEN}}
    });

    return model;
}

StatefulLLMPipeline build_pipeline_for_npu(const std::filesystem::path& model_path,
                                           const Tokenizer& tokenizer,
                                           const std::string& device) {
    auto properties = configure_npu_properties(device);
    auto model = load_model_for_npu(model_path / "openvino_model.xml", properties);

    GenerationConfig gen_config;
    gen_config.max_length = MAX_PROMPT_LEN + MIN_RESPONSE_LEN;
    gen_config.eos_token_id = tokenizer.get_eos_token_id();
    gen_config.num_return_sequences = 1;
    gen_config.do_sample = false; // greedy decoding 기본값

    return StatefulLLMPipeline(model, tokenizer, device, properties, gen_config);
}

int main() {
    std::string model_dir = "./Llama-2-7B-Chat-FP16";  // GPTQ 말고 FP16으로 export한 모델 디렉토리
    std::string device = "NPU";

    Tokenizer tokenizer(model_dir);
    auto pipeline = build_pipeline_for_npu(model_dir, tokenizer, device);

    std::string prompt = "What is the capital of France?";
    auto result = pipeline.generate(prompt);

    for (const auto& output : result.texts) {
        std::cout << "\n[Answer]: " << output << std::endl;
    }

    return 0;
}
