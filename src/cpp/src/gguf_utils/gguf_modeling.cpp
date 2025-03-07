#include <iostream>

#include "gguf.h"
#include "gguf_modeling.h"

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path) {
    auto gguf_data = load_gguf(model_path);
    auto& weights = gguf_data.first;
    auto& metadata = gguf_data.second;

    if (metadata.find("general.architecture") != metadata.end()) {
        std::cout << std::get<std::string>(metadata["general.architecture"]);
    }
}
