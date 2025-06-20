// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/visibility.hpp"

#ifdef openvino_genai_EXPORTS
#    define OPENVINO_GENAI_EXPORTS OPENVINO_CORE_EXPORTS
#else
#    define OPENVINO_GENAI_EXPORTS OPENVINO_CORE_IMPORTS
#endif  // openvino_genai_EXPORTS
