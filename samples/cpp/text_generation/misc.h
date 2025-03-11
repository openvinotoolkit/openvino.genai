// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef MISC_H
#define MISC_H

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <ctime>

#ifdef DEBUG
#define DEBUG_PRINT(x) std::cout << "[GenAI_Andorid:] " << x << std::endl
#else
#define DEBUG_PRINT(x)
#endif

typedef struct _mem_info {
    unsigned long rss_mem;
    unsigned long sh_mem;
    unsigned long uss_mem;
} MEM_CONSUME_INFO;

pthread_t init_memory_consumption();
std::tm* gettimenow();
std::string gettimestamp(std::tm* timeinfo);
int getpromptfromfile(std::string file_path, std::string& prompt_s, int* token_size);

#endif  // MISC_H
