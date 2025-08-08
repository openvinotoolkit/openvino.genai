
#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"
extern "C" {
    #include "safetensors.h"
}

using namespace ov::op;
// Converts Safetensors element type to OV element type. Only part of the types are supported.
ov::element::Type safetensors_to_ov_element_type (int dtype);

using ConstantMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

// Safetensor file parser that deallocates temporary buffers automatically.
// Drop-in replacement for the third party safetensors_File struct.
struct AutoSafetensor: public safetensors_File {
    ~AutoSafetensor () {
        std::free(tensors);
        std::free(metadata);
    }
};

// The key in the map is a tensor name and the Constant uses a region of memory from the memory block.
// Each Constant holds a shared pointer to the block in the runtime info.
// The memory block will be deallocated when the last Constant is destroyed.
ConstantMap safetensor_to_constant_map(const ov::Tensor& safetensor);

// Reads a file with a given filename expecting Safetensors file format.
// The file data is mmaped to tensor.
ConstantMap read_safetensors(const std::filesystem::path& filename);