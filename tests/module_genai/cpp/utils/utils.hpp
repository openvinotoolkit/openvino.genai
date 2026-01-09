
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

bool readFileToString(const std::string &filename, std::string &content);

// Get the absolute path to the data directory.
// It first checks the DATA_DIR environment variable.
// If the variable is not set, it uses the default path "./test_data".
std::string get_data_path();

// Get the absolute path to the model directory.
// It first checks the MODEL_DIR environment variable.
// If the variable is not set, it uses the default path "./test_models".
std::string get_model_path();

bool check_env_variable(const std::string& var_name);

bool check_file_exists(const std::string& path);