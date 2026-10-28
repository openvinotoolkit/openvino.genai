// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "llama.h"

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    auto params = llama_model_quantize_default_params();
    const std::string type = argv[3];
    if (type == "F16")
        params.ftype = LLAMA_FTYPE_MOSTLY_F16;
    else if (type == "Q4_0")
        params.ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
    else
        return 2;
    params.nthread = 4;
    params.pure = true;
    params.allow_requantize = true;
    return llama_model_quantize(argv[1], argv[2], &params);
}
