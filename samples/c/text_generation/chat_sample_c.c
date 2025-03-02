// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/llm_pipeline_c.h"

#define MAX_PROMPT_LENGTH 64
#define MAX_OUTPUT_LENGTH 1024

#ifdef _WIN32
#    include <windows.h>
#    define THREAD_RETURN                      DWORD WINAPI      // Thread return type
#    define THREAD_HANDLE                      HANDLE            // Thread handle
#    define MUTEX_TYPE                         CRITICAL_SECTION  // Mutex type
#    define INIT_MUTEX(m)                      InitializeCriticalSection(&m)
#    define LOCK_MUTEX(m)                      EnterCriticalSection(&m)
#    define UNLOCK_MUTEX(m)                    LeaveCriticalSection(&m)
#    define DESTROY_MUTEX(m)                   DeleteCriticalSection(&m)
#    define CREATE_THREAD(thread, func, param) thread = CreateThread(NULL, 0, func, param, 0, NULL)
#    define SLEEP(ms)                          Sleep(ms)
#    define JOIN_THREAD(thread)                WaitForSingleObject(thread, INFINITE)
#else
#    include <pthread.h>
#    include <unistd.h>
#    define THREAD_RETURN                      void*            // Thread return type
#    define THREAD_HANDLE                      pthread_t        // Thread handle
#    define MUTEX_TYPE                         pthread_mutex_t  // Mutex type
#    define INIT_MUTEX(m)                      pthread_mutex_init(&m, NULL)
#    define LOCK_MUTEX(m)                      pthread_mutex_lock(&m)
#    define UNLOCK_MUTEX(m)                    pthread_mutex_unlock(&m)
#    define DESTROY_MUTEX(m)                   pthread_mutex_destroy(&m)
#    define CREATE_THREAD(thread, func, param) pthread_create(&thread, NULL, func, param)
#    define SLEEP(ms)                          usleep((ms) * 1000)
#    define JOIN_THREAD(thread)                pthread_join(thread, NULL)
#endif

#define BUFFER_SIZE 1024

char buffer[BUFFER_SIZE];  // Stream buffer
int buffer_pos = 0;        // Current position in the buffer
size_t last_pos = 0;       // Last read position
MUTEX_TYPE buffer_lock;    // Mutex lock
volatile int running = 1;  // Thread running flag

// Listener thread, periodically checks the buffer and outputs
THREAD_RETURN listen_buffer(void* param) {
    while (running) {
        SLEEP(100);

        size_t current_pos = buffer_pos;  // Read buffer_pos without locking
        if (current_pos > last_pos) {     // Print only if there is new data
            LOCK_MUTEX(buffer_lock);
            fwrite(buffer + last_pos, 1, current_pos - last_pos, stdout);
            fflush(stdout);

            last_pos = current_pos;
            UNLOCK_MUTEX(buffer_lock);
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <MODEL_DIR>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char prompt[MAX_PROMPT_LENGTH], output[MAX_OUTPUT_LENGTH];
    const char* models_path = argv[1];
    const char* device = "CPU";  // GPU, NPU can be used as well
    ov_genai_llm_pipeline* pipeline = ov_genai_llm_pipeline_create(models_path, "CPU");
    if (pipeline == NULL) {
        fprintf(stderr, "Failed to create LLM pipeline\n");
        return EXIT_FAILURE;
    }

    ov_genai_generation_config* config = ov_genai_generation_config_create();
    ov_genai_generation_config_set_max_new_tokens(config, BUFFER_SIZE);

    ov_genai_llm_pipeline_start_chat(pipeline);
    INIT_MUTEX(buffer_lock);
    THREAD_HANDLE listener_thread;
    CREATE_THREAD(listener_thread, listen_buffer, NULL);
    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        prompt[strcspn(prompt, "\n")] = 0;

        ov_genai_llm_pipeline_generate_stream(pipeline, prompt, output, sizeof(output), config, buffer, BUFFER_SIZE, &buffer_pos);
        SLEEP(3000); // Sleep to allow the listener thread to flush the buffer.

        LOCK_MUTEX(buffer_lock);
        memset(buffer, 0, BUFFER_SIZE);
        memset(prompt, 0, MAX_PROMPT_LENGTH);
        buffer_pos = 0;
        last_pos = 0;
        UNLOCK_MUTEX(buffer_lock);
        printf("\n----------\nquestion:\n");
    }
    ov_genai_llm_pipeline_finish_chat(pipeline);
    ov_genai_llm_pipeline_free(pipeline);
    ov_genai_generation_config_free(config);
    running = 0;
    JOIN_THREAD(listener_thread);

    return EXIT_SUCCESS;
}
