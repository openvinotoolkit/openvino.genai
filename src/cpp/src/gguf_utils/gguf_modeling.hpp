#pragma once

#include <cstring>

#include "openvino/openvino.hpp"

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path);