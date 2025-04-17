#pragma once

#include <cstring>

#include "openvino/openvino.hpp"
#include "openvino/genai/visibility.hpp"

std::shared_ptr<ov::Model> OPENVINO_GENAI_EXPORTS create_from_gguf(const std::string& model_path);