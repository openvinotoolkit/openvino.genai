// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/visibility.hpp"

#ifdef genai_EXPORTS
#    define OPENVINO_GENAI_EXPORTS OPENVINO_CORE_EXPORTS
#else
#    define OPENVINO_GENAI_EXPORTS OPENVINO_CORE_IMPORTS
#endif  // genai_EXPORTS
