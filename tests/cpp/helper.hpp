// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers);