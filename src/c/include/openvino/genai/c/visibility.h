// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef OPENVINO_GENAI_EXTERN_C
#    ifdef __cplusplus
#        define OPENVINO_GENAI_EXTERN_C extern "C"
#    else
#        define OPENVINO_GENAI_EXTERN_C
#    endif
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#    ifdef openvino_genai_c_EXPORTS
#        define OPENVINO_GENAI_C_EXPORTS OPENVINO_GENAI_EXTERN_C __declspec(dllexport)
#    else
#        define OPENVINO_GENAI_C_EXPORTS OPENVINO_GENAI_EXTERN_C __declspec(dllimport)
#    endif
#elif defined(__GNUC__) && (__GNUC__ >= 4) || defined(__clang__)
#    ifdef openvino_genai_c_EXPORTS
#        define OPENVINO_GENAI_C_EXPORTS OPENVINO_GENAI_EXTERN_C __attribute__((visibility("default")))
#    else
#        define OPENVINO_GENAI_C_EXPORTS OPENVINO_GENAI_EXTERN_C __attribute__((visibility("default")))
#    endif
#else
#    define OPENVINO_GENAI_C_EXPORTS OPENVINO_GENAI_EXTERN_C
#endif
