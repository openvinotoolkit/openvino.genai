// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "misc.h"
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

MEM_CONSUME_INFO genai_mem_info;
bool exit_mem_read = false;

std::tm* gettimenow() {
    std::time_t now = std::time(nullptr);  // Get current time
    static std::tm timenow;
    timenow = {0};
    localtime_r(&now, &timenow);  // Conver to local time

    return &timenow;
}

std::string gettimestamp(std::tm* timeinfo) {
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);  // Format timestamp

    return std::string(buffer);
}

int getpromptfromfile(std::string file_path, std::string& prompt_s, int* token_size) {

    std::ifstream infile(file_path);
    if (!infile.is_open()) {
        std::cout << "Failed to Prompt open file " << file_path << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    bool prompt = false, prompt_len = false;
    int lines = 1;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        try {
            json j = json::parse(line);
            for (auto& [key, value] : j.items()) {
                if (key == "prompt") {
                    prompt_s = value;
                    prompt = true;
                }
                if (key == "token_size") {
                    *token_size = std::stoi(std::string(value));
                    prompt_len = true;
                }
            }

        } catch (const json::parse_error& e) {
            std::cout << "Parse error on line " << lines << ": " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        ++lines;
    }
    infile.close();

    if (!prompt) {
        std::cout << "Failed to locate Prompt sentence in prompt file: " << file_path << std::endl;
        return EXIT_FAILURE;
    }
    // TBD: change this from str.length() to word count
    if (!prompt_len) *token_size = prompt_s.length();

    return EXIT_SUCCESS;
}

static void get_memory_consumption_info(unsigned long* rss_mem, unsigned long* sh_mem,
                                        unsigned long* uss_mem) {
    pid_t pid = getpid();
    std::string info;
    std::string statpath = "/proc/";
    unsigned long vmrss, shmem, usmem;
    bool bvmrss, bshmem;

    statpath += std::to_string(pid) + "/status";
    std::ifstream file(statpath);
    while (!exit_mem_read) {
        bvmrss = true;
        bshmem = true;
        while (std::getline(file, info)) {
            if (bvmrss && (info.rfind("VmRSS:", 0) == 0)) {  // Line starts with "VmRSS"
                std::string value;
                std::istringstream iss(info);
                iss.ignore(256, ':');      // Ignore until : of 'VmRSS:'
                iss >> value;              // Extract memory size
                vmrss = std::stol(value);  // Convert to long
                bvmrss = false;
            }
            if (bshmem && (info.rfind("RssShmem:", 0) == 0)) {
                std::string value;
                std::istringstream iss(info);
                iss.ignore(256, ':');
                iss >> value;
                shmem = std::stol(value);
                bshmem = false;
            }
        }

        usmem = vmrss - shmem;
        if (vmrss > *rss_mem) *rss_mem = vmrss;
        if (shmem > *sh_mem) *sh_mem = shmem;
        if (usmem > *uss_mem) *uss_mem = usmem;

        // Take file ptr to start and read again from start
        file.clear();
        file.seekg(0, std::ios::beg);
        usleep(1000);
    }
    std::cout<<"exiting the thread"<<std::endl;
}

static void* memory_thread(void* params) {
    get_memory_consumption_info(&genai_mem_info.rss_mem, &genai_mem_info.sh_mem,
                                &genai_mem_info.uss_mem);
}

pthread_t init_memory_consumption() {
    pthread_t thread;
    int tnum;

    if (pthread_create(&thread, NULL, &memory_thread, &tnum) != 0) {
        std::cout << "Error in Memory Consumption thread creation" << std::endl;
        return errno;
    }

    return thread;
}
