#include <stdio.h>
#include <stdlib.h>

#include "openvino/genai/openvino_genai_c.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<PROMPT>\"\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* model_dir = argv[1];
    const char* prompt = argv[2];

    LLMPipelineHandle* pipeline = CreateLLMPipeline(model_dir, "CPU");
    if (pipeline == NULL) {
        fprintf(stderr, "Failed to create LLM pipeline\n");
        return EXIT_FAILURE;
    }
    GenerationConfigHandle* config = CreateGenerationConfig();
    GenerationConfigSetMaxNewTokens(config, 100);

    char output[1024];
    LLMPipelineGenerate(pipeline, prompt, output, sizeof(output), config);

    printf("Generated text: %s\n", output);

    DestroyLLMPipeline(pipeline);
    DestroyGenerationConfig(config);

    return EXIT_SUCCESS;
}
