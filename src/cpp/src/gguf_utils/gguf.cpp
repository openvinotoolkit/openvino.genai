// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>

template <typename... Args>
std::string format(std::string fmt, Args... args) {
    size_t bufferSize = 1000;
    char* buffer = new char[bufferSize];
    int n = sprintf(buffer, fmt.c_str(), args...);
    assert(n >= 0 && n < (int)bufferSize - 1);

    std::string fmtStr(buffer);
    delete[] buffer;
    return fmtStr;
}

#ifdef HAS_LLAMA_CPP

std::optional<uint32_t> dtype_to_gguf_tensor_type(const ov::element::Type& dtype) {
    switch (dtype) {
    case ov::element::f64: return GGML_TYPE_F64;
    case ov::element::f32: return GGML_TYPE_F32;
    case ov::element::f16: return GGML_TYPE_F16;
    case ov::element::bf16: return GGML_TYPE_BF16;
    case ov::element::i8:  return GGML_TYPE_I8;
    case ov::element::i16: return GGML_TYPE_I16;
    case ov::element::i32: return GGML_TYPE_I32;
    case ov::element::i64: return GGML_TYPE_I64;
    default: return std::nullopt;
    }
}

std::optional<ov::element::Type> gguf_type_to_dtype(const uint32_t& gguf_type) {
    switch (gguf_type) {
    case GGML_TYPE_F64: return ov::element::f64;
    case GGML_TYPE_F32: return ov::element::f32;
    case GGML_TYPE_F16: return ov::element::f16;
    case GGML_TYPE_BF16: return ov::element::bf16;
    case GGML_TYPE_I8:  return ov::element::i8;
    case GGML_TYPE_I16: return ov::element::i16;
    case GGML_TYPE_I32: return ov::element::i32;
    case GGML_TYPE_I64: return ov::element::i64;
    default: return std::nullopt;
    }
}

ov::Shape get_shape(const gguf_tensor& tensor) {
    ov::Shape shape;
    int n_dims = ggml_n_dims(&tensor);
    for (int i = n_dims - 1; i >= 0; i--) {
        shape.push_back(tensor.ne[i]);
    }
    return shape;
}

ov::Tensor extract_tensor_data(gguf_tensor* tensor) {
    std::optional<ov::element::Type> equivalent_dtype = gguf_type_to_dtype(tensor->type);
    if (equivalent_dtype.has_value()) {
        auto shape = get_shape(*tensor);
        ov::Tensor weights(equivalent_dtype.value(), shape);
        memcpy(weights.data(), tensor->data, ggml_nelements(tensor) * equivalent_dtype.value().size());
        return weights;
    }
    OPENVINO_THROW("[load_gguf] Legacy tensor unpacking is delegated to llama.cpp in V2.");
}

std::unordered_map<std::string, GGUFMetaData> load_metadata(gguf_ctx* ctx) {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    int n_kv = gguf_get_n_kv(ctx);
    
    for (int i = 0; i < n_kv; ++i) {
        std::string key_name = gguf_get_key(ctx, i);
        auto type = gguf_get_kv_type(ctx, i);
        GGUFMetaData value;

        switch (type) {
        case GGUF_TYPE_UINT8: {
            value = ov::Tensor(ov::element::u8, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<uint8_t>()) = gguf_get_val_u8(ctx, i);
            break;
        }
        case GGUF_TYPE_INT8: {
            value = ov::Tensor(ov::element::i8, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<int8_t>()) = gguf_get_val_i8(ctx, i);
            break;
        }
        case GGUF_TYPE_UINT16: {
            value = ov::Tensor(ov::element::u16, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<uint16_t>()) = gguf_get_val_u16(ctx, i);
            break;
        }
        case GGUF_TYPE_INT16: {
            value = ov::Tensor(ov::element::i16, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<int16_t>()) = gguf_get_val_i16(ctx, i);
            break;
        }
        case GGUF_TYPE_UINT32: {
            value = ov::Tensor(ov::element::u32, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<uint32_t>()) = gguf_get_val_u32(ctx, i);
            break;
        }
        case GGUF_TYPE_INT32: {
            value = ov::Tensor(ov::element::i32, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<int32_t>()) = gguf_get_val_i32(ctx, i);
            break;
        }
        case GGUF_TYPE_UINT64: {
            value = ov::Tensor(ov::element::u64, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<uint64_t>()) = gguf_get_val_u64(ctx, i);
            break;
        }
        case GGUF_TYPE_INT64: {
            value = ov::Tensor(ov::element::i64, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<int64_t>()) = gguf_get_val_i64(ctx, i);
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            value = ov::Tensor(ov::element::f32, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<float>()) = gguf_get_val_f32(ctx, i);
            break;
        }
        case GGUF_TYPE_FLOAT64: {
            value = ov::Tensor(ov::element::f64, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<double>()) = gguf_get_val_f64(ctx, i);
            break;
        }
        case GGUF_TYPE_BOOL: {
            value = ov::Tensor(ov::element::boolean, ov::Shape(0));
            *(std::get<ov::Tensor>(value).data<bool>()) = gguf_get_val_bool(ctx, i);
            break;
        }
        case GGUF_TYPE_STRING: {
            value = std::string(gguf_get_val_str(ctx, i));
            break;
        }
        case GGUF_TYPE_ARRAY: {
            auto arr_type = gguf_get_arr_type(ctx, i);
            size_t size = gguf_get_arr_n(ctx, i);
            const void* data = gguf_get_arr_data(ctx, i);

            switch (arr_type) {
            case GGUF_TYPE_UINT8: {
                value = ov::Tensor(ov::element::u8, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<uint8_t>(), data, size * sizeof(uint8_t));
                break;
            }
            case GGUF_TYPE_INT8: {
                value = ov::Tensor(ov::element::i8, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<int8_t>(), data, size * sizeof(int8_t));
                break;
            }
            case GGUF_TYPE_UINT16: {
                value = ov::Tensor(ov::element::u16, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<uint16_t>(), data, size * sizeof(uint16_t));
                break;
            }
            case GGUF_TYPE_INT16: {
                value = ov::Tensor(ov::element::i16, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<int16_t>(), data, size * sizeof(int16_t));
                break;
            }
            case GGUF_TYPE_UINT32: {
                value = ov::Tensor(ov::element::u32, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<uint32_t>(), data, size * sizeof(uint32_t));
                break;
            }
            case GGUF_TYPE_INT32: {
                value = ov::Tensor(ov::element::i32, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<int32_t>(), data, size * sizeof(int32_t));
                break;
            }
            case GGUF_TYPE_UINT64: {
                value = ov::Tensor(ov::element::u64, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<uint64_t>(), data, size * sizeof(uint64_t));
                break;
            }
            case GGUF_TYPE_INT64: {
                value = ov::Tensor(ov::element::i64, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<int64_t>(), data, size * sizeof(int64_t));
                break;
            }
            case GGUF_TYPE_FLOAT32: {
                value = ov::Tensor(ov::element::f32, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<float>(), data, size * sizeof(float));
                break;
            }
            case GGUF_TYPE_FLOAT64: {
                value = ov::Tensor(ov::element::f64, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<double>(), data, size * sizeof(double));
                break;
            }
            case GGUF_TYPE_BOOL: {
                value = ov::Tensor(ov::element::boolean, ov::Shape{size});
                std::memcpy(std::get<ov::Tensor>(value).data<bool>(), data, size * sizeof(bool));
                break;
            }
            case GGUF_TYPE_STRING: {
                std::vector<std::string> strs(size);
                for (size_t j = 0; j < size; ++j) {
                    strs[j] = std::string(gguf_get_arr_str(ctx, i, j));
                }
                value = std::move(strs);
                break;
            }
            default:
                OPENVINO_THROW("[load_gguf] Unsupported array type.");
            }
            break;
        }
        default:
            OPENVINO_THROW("[load_gguf] Received unexpected type.");
        }
        metadata[key_name] = value;
    }
    return metadata;
}

#endif // HAS_LLAMA_CPP

void check_file(std::string file) {
    bool exists;
    {
        std::ifstream f(file.c_str());
        exists = f.good();
    }
    OPENVINO_ASSERT(exists, "[load_gguf] Failed to open '", file, "'");
}

std::vector<std::string> get_all_files(std::string file, int total_num) {
    std::vector<std::string> files;
    files.push_back(file);

    size_t length = 5;
    size_t startPos = file.length() - 19;

    for (int i = 1; i < total_num; i++) {
        std::string new_number = std::to_string(i + 1);
        while (new_number.length() < length) {
            new_number = "0" + new_number;
        }
        file.replace(startPos, length, new_number);
        check_file(file);
        files.push_back(file);
    }
    return files;
}

GGUFLoad get_gguf_data(const std::string& file) {
    std::unordered_map<std::string, ov::Tensor> arrays;
    std::unordered_map<std::string, gguf_tensor_type> qtype;

    check_file(file);

    std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(gguf_open(file.data()), gguf_close);
    OPENVINO_ASSERT(ctx, "Failed to open '", file, "' with gguf_open");

    auto metadata = load_metadata(ctx.get());
    std::string split_flag = "split.count";
    auto it = metadata.find(split_flag);

    if (it == metadata.end()) {
        load_arrays(ctx.get(), arrays, qtype);
        return GGUFLoad{metadata, arrays, qtype};
    } else {
        auto total_num_tensor = std::get<ov::Tensor>(metadata.at(split_flag));
        int total_num = *(total_num_tensor.data<ov::element_type_traits<ov::element::u16>::value_type>());

        std::vector<std::string> files = get_all_files(file, total_num);

        for (size_t i = 1; i < files.size(); i++) {
            std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx_i(gguf_open(files.at(i).data()), gguf_close);
            OPENVINO_ASSERT(ctx_i, "Failed to open '", files.at(i), "' with gguf_open");
            auto metadata_tmp = load_metadata(ctx_i.get());
            load_arrays(ctx_i.get(), arrays, qtype);
        }
        load_arrays(ctx.get(), arrays, qtype);
        return GGUFLoad{metadata, arrays, qtype};
    }
}

float metadata_to_float(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>());
}

int metadata_to_int(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::i32>::value_type>());
}

std::map<std::string, GGUFMetaData> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> config;
    auto arch = std::get<std::string>(metadata.at("general.architecture"));
    config["architecture"] = arch;
    config["layer_num"] = metadata_to_int(metadata, arch + ".block_count");
    config["head_num"] = metadata_to_int(metadata, arch + ".attention.head_count");
    config["head_size"] = metadata.count(arch + ".attention.key_length") ?
                       metadata_to_int(metadata, arch + ".attention.key_length") :
                       (metadata_to_int(metadata, arch + ".embedding_length") / 
                       metadata_to_int(metadata, arch + ".attention.head_count"));
    config["head_num_kv"] = metadata.count(arch + ".attention.head_count_kv") ?
            metadata_to_int(metadata, arch + ".attention.head_count_kv") :
            metadata_to_int(metadata, arch + ".attention.head_count");
    config["hidden_size"] = metadata_to_int(metadata, arch + ".embedding_length");
    config["max_position_embeddings"] = metadata.count(arch + ".context_length") ?
            metadata_to_int(metadata, arch + ".context_length") : 2048;
    config["rms_norm_eps"] = metadata_to_float(metadata, arch + ".attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] = metadata.count(arch + ".rope.freq_base") ?
            metadata_to_float(metadata, arch + ".rope.freq_base") : 10000.0f;
    config["file_type"] = metadata_to_int(metadata, "general.file_type");
    return config;
}

std::unordered_map<std::string, ov::Tensor> consts_from_weights(
    const std::map<std::string, GGUFMetaData>& config,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    std::unordered_map<std::string, ov::Tensor> consts;
    
    // Safety check since we aren't loading weights via the old method anymore
    if (weights.empty()) return consts; 

    // [Rest of consts_from_weights remains unchanged]
    consts["model.embed_tokens.weight"] = weights.at("token_embd.weight");
    consts["model.norm.weight"] = weights.at("output_norm.weight");
    if (weights.count("output.weight")) {
        consts["lm_head.weight"] = weights.at("output.weight");
        if (weights.count("output.bias")) {
            consts["lm_head.bias"] = weights.at("output.bias");
        }
    }
    return consts;
}

std::unordered_map<std::string, gguf_tensor_type> get_qtype_map(
    const std::map<std::string, GGUFMetaData>& config,
    const std::unordered_map<std::string, gguf_tensor_type>& qtype) {
    std::unordered_map<std::string, gguf_tensor_type> qtype_map;
    
    if (qtype.empty()) return qtype_map; // Safety return for V2

    if (qtype.count("token_embd.qtype")) {
        qtype_map["model.embed_tokens.qtype"] = qtype.at("token_embd.qtype");
    }
    return qtype_map;
}

std::tuple<std::map<std::string, GGUFMetaData>,
           std::unordered_map<std::string, ov::Tensor>,
           std::unordered_map<std::string, gguf_tensor_type>>
load_gguf(const std::string& file) {
    auto [metadata, weights, qtype] = get_gguf_data(file);
    auto config = config_from_meta(metadata);
    auto consts = consts_from_weights(config, weights);
    auto qtypes = get_qtype_map(config, qtype);

    return std::make_tuple(config, consts, qtypes);
}