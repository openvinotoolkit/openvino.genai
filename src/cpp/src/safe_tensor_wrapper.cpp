#include "safe_tensor_wrapper.hpp"

ov::element::Type safetensors_to_ov_element_type (int dtype) {
    switch(dtype) {
        case SAFETENSORS_F32:
            return ov::element::f32;
        case SAFETENSORS_F16:
            return ov::element::f16;
        case SAFETENSORS_BF16:
            return ov::element::bf16;
        case SAFETENSORS_I64:
            return ov::element::i64;
        case SAFETENSORS_BOOL:
            return ov::element::boolean;
        default:
            OPENVINO_THROW("Not supported safetensors dtype: ", dtype);
    }
}

ConstantMap safetensor_to_constant_map(const ov::Tensor& safetensor) {
    AutoSafetensor safe_tensors_file{};

    OPENVINO_ASSERT(safetensors_file_init(safetensor.data<char>(), safetensor.get_byte_size(), &safe_tensors_file) == nullptr,
        "Cannot parse safetensor as a Safetensors file format. Safetensors file format is supported only"
    );

    ConstantMap tensors;
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        safetensors_TensorDescriptor tensor = safe_tensors_file.tensors[i];
        std::string name(tensor.name.ptr, tensor.name.ptr + tensor.name.len);
        ov::Shape shape(tensor.shape, tensor.shape + tensor.n_dimensions);
        void* ptr = tensor.ptr;     // FIXME: needs a non-constant pointer because Tensor doesn't accept a constant pointer

        auto type = safetensors_to_ov_element_type(tensor.dtype);
        auto constant =
            std::make_shared<v0::Constant>(type, shape, ptr, nullptr);      // wraps existing memory, no ownership
        constant->get_rt_info()["__safetensors_buffer_holder"] = safetensor;    // to automatically deallocate underlying memory buffer when last constant that holds it is destroyed
        tensors[name] = constant;
    }
    return tensors;
}

ConstantMap read_safetensors(const std::filesystem::path& filename) {
    auto safetensor = ov::read_tensor_data(filename);

    return safetensor_to_constant_map(safetensor);
}
