// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>

#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/text2speech_pipeline.h"
#include "text2speech_sample_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> \"<PROMPT>\" [<SPEAKER_EMBEDDING_BIN_FILE>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_dir = argv[1];
    const char* prompt = argv[2];
    const char* speaker_embedding_path = (argc == 4) ? argv[3] : NULL;
    const char* device = "CPU";
    int exit_status = EXIT_SUCCESS;

    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* speaker_embedding = NULL;

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_dir, device, 0, &pipeline));

    if (speaker_embedding_path) {
        speaker_embedding = read_speaker_embedding(speaker_embedding_path);
        if (!speaker_embedding) {
            exit_status = EXIT_FAILURE;
            goto err;
        }
    }

    const char* texts[] = {prompt};
    CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, texts, 1, speaker_embedding, 0, &results));

    size_t count = 0;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &count));
    if (count != 1) {
        fprintf(stderr, "Expected exactly one decoded waveform\n");
        exit_status = EXIT_FAILURE;
        goto err;
    }

    ov_tensor_t* speech_tensor = NULL;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech_tensor));

    void* waveform_data = NULL;
    ov_tensor_data(speech_tensor, &waveform_data);

    ov_shape_t shape;
    ov_tensor_get_shape(speech_tensor, &shape);
    size_t waveform_size = 1;
    for (size_t i = 0; i < shape.rank; ++i)
        waveform_size *= shape.dims[i];
    ov_shape_free(&shape);

    const char* output_file = "output_audio.wav";
    save_to_wav((const float*)waveform_data, waveform_size, output_file);
    printf("[Info] Text successfully converted to audio file \"%s\".\n", output_file);

    ov_tensor_free(speech_tensor);

err:
    if (results)
        ov_genai_text2speech_decoded_results_free(results);
    if (pipeline)
        ov_genai_text2speech_pipeline_free(pipeline);
    if (speaker_embedding)
        ov_tensor_free(speaker_embedding);

    return exit_status;
}
