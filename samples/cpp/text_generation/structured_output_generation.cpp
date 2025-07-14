// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 2 || argc > 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    std::string prompt;
    std::string models_path = argv[1];

    // Default device is CPU; can be overridden by the second argument
    std::string device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 1000;
    config.do_sample = false;

    std::string json_schema = R"({
        "$defs": {
            "Step": {
                "properties": {
                    "explanation": {
                        "title": "Explanation",
                        "type": "string"
                    },
                    "output": {
                        "title": "Output",
                        "type": "string"
                    }
                },
                "required": ["explanation", "output"],
                "title": "Step",
                "type": "object"
            }
        },
        "properties": {
            "steps": {
                "items": {
                    "$ref": "#/$defs/Step"
                },
                "title": "Steps",
                "type": "array"
            },
            "final_answer": {
                "title": "Final Answer",
                "type": "string"
            }
        },
        "required": ["steps", "final_answer"],
        "title": "MathReasoning",
        "type": "object"
    })";
    config.structured_output_config = ov::genai::StructuredOutputConfig(
        ov::AnyMap{{ov::genai::json_schema(json_schema)}}
    );

    std::string sys_message = R"(
    Decompose the task and do it step by step and include it in a structured JSON.
    For every mathematical equation use the adequate mathematical method. Do not try to solve linear equations
    as a quadratic/cubic ones and vice versa.
    For example for 2*x - x**2 + 15 = 0 the output format should be as the following: 
    {"steps": [
        {"explanation": "Rearranging the equation to isolate x.", "output": "2*x - x**2 + 15 = 0"},
        {"explanation": "Rearranging the equation to standard form.", "output": "-x**2 + 2*x + 15 = 0"},
        {"explanation": "Factoring the quadratic equation.", "output": "-(x - 5)(x + 3) = 0"},
        {"explanation": "Setting each factor to zero to find the roots.", "output": "x - 5 = 0 or x + 3 = 0"},
        {"explanation": "Finding the solutions for x.", "output": "x = 5 or x = -3"}
    ], "final_answer": "x = 5 or x = -3"}.
    "output" field should contain only mathematical notations without text.
    )";
    
    auto streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        return ov::genai::StreamingStatus::RUNNING;
    };

    pipe.start_chat(sys_message);
    std::cout << "This is a sample of structured output generation.\n"
              << "You can enter a mathematical equation, and the model will solve it step by step.\n"
              << "For example, try: 2*x -2 + 15 = 0\n"
              << "To exit, press Ctrl+C or close the terminal.\n"
              << "> ";

    while (std::getline(std::cin, prompt)) {
        pipe.generate(prompt, config, streamer);
        std::cout << "\n----------\n"
            "> ";
    }
    pipe.finish_chat();
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
