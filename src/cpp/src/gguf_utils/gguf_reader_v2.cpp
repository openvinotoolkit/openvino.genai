#include "gguf_reader_v2.hpp"
#include <openvino/openvino.hpp>

// Our new bridged headers!
#ifdef HAS_LLAMA_CPP
#include "llama.h"
#include "ggml.h"
#endif

#include <iostream>
#include <stdexcept>

namespace ov {
namespace genai {

// Constructor
GGUFReaderV2::GGUFReaderV2() {
#ifdef HAS_LLAMA_CPP
    // Initialize the llama.cpp backend when the reader is created
    llama_backend_init();
#else
    throw std::runtime_error("GenAI was built without llama.cpp support!");
#endif
}

// Destructor
GGUFReaderV2::~GGUFReaderV2() {
#ifdef HAS_LLAMA_CPP
    // Clean up memory
    llama_backend_free();
#endif
}

// The Main Read Function
std::shared_ptr<ov::Model> GGUFReaderV2::read(const std::string& filename) {
#ifndef HAS_LLAMA_CPP
    throw std::runtime_error("Cannot read GGUF: llama.cpp backend is disabled.");
#else
    std::cout << "[GGUF V2] Attempting to open: " << filename << std::endl;

    // 1. Set up default parameters for loading the model
    llama_model_params model_params = llama_model_default_params();
    
    // We only want to read the weights, not run inference yet, 
    // so we can tell llama.cpp to keep it entirely on the CPU for now.
    model_params.n_gpu_layers = 0; 

    // 2. Load the GGUF file into the llama_model struct
    llama_model* model = llama_load_model_from_file(filename.c_str(), model_params);
    
    if (model == nullptr) {
        throw std::runtime_error("Failed to load GGUF file via llama.cpp!");
    }

    // 3. Let's just print some basic info to prove it worked
    // (We will replace this with actual OpenVINO conversion logic next)
    std::cout << "[GGUF V2] Successfully loaded model!" << std::endl;
    // Note: The specific function to get tensor count depends slightly on the llama.cpp version,
    // but we can start by just acknowledging the load was successful.

    // 4. Free the model memory for this test phase
    llama_free_model(model);

    // Return a dummy empty model for now just so it compiles
    return std::make_shared<ov::Model>(ov::NodeVector{}, ov::ParameterVector{});
#endif
}

} // namespace genai
} // namespace ov