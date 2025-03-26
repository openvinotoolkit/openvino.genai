#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>

#include "gguf.hpp"

#include <iostream>

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

using GGUFLoad = std::pair<
    std::unordered_map<std::string, GGUFMetaData>,
    std::unordered_map<std::string, ov::Tensor>>;

template<typename... Args>
std::string format(std::string fmt, Args... args)
{
    size_t bufferSize = 1000;
    char *buffer = new char[bufferSize];
    int n = sprintf(buffer, fmt.c_str(), args...);
    assert (n >= 0 and n < (int) bufferSize - 1  && "check fmt_str output");

    std::string fmtStr (buffer);
    delete buffer;
    return fmtStr;
}

std::optional<uint32_t> dtype_to_gguf_tensor_type(const ov::element::Type& dtype) {
  switch (dtype) {
    case ov::element::f32:
      return GGUF_TYPE_F32;
    case ov::element::f16:
      return GGUF_TYPE_F16;
    case ov::element::i8:
      return GGUF_TYPE_I8;
    case ov::element::i16:
      return GGUF_TYPE_I16;
    case ov::element::i32:
      return GGUF_TYPE_I32;
    default:
      return std::nullopt;
  }
}

std::optional<ov::element::Type> gguf_type_to_dtype(const uint32_t& gguf_type) {
  switch (gguf_type) {
    case GGUF_TYPE_F32:
      return ov::element::f32;
    case GGUF_TYPE_F16:
      return ov::element::f16;
    case GGUF_TYPE_I8:
      return ov::element::i8;
    case GGUF_TYPE_I16:
      return ov::element::i16;
    case GGUF_TYPE_I32:
      return ov::element::i32;
    default:
      return std::nullopt;
  }
}

ov::Shape get_shape(const gguf_tensor& tensor) {
  ov::Shape shape;
  // The dimension order in GGML is the reverse of the order used in MLX.
  for (int i = tensor.ndim - 1; i >= 0; i--) {
    shape.push_back(tensor.dim[i]);
  }
  return shape;
}

ov::Tensor extract_tensor_data(gguf_tensor* tensor) {
  std::optional<ov::element::Type> equivalent_dtype = gguf_type_to_dtype(tensor->type);
  // If there's an equivalent type, we can simply copy.
  if (equivalent_dtype.has_value()) {
    auto shape = get_shape(*tensor);
    ov::Tensor weights(equivalent_dtype.value(), shape);

    memcpy(
        weights.data(),
        tensor->weights_data,
        tensor->num_weights * equivalent_dtype.value().size());
    return weights;
  }
  // Otherwise, we convert to float16.
  // TODO: Add other dequantization options.
  int16_t* data = gguf_tensor_to_f16(tensor);
  auto shape = get_shape(*tensor);
  if (data == NULL) {
    throw std::runtime_error("[load_gguf] gguf_tensor_to_f16 failed");
  }
  const size_t new_size = tensor->num_weights * sizeof(int16_t);
  ov::Tensor weights(ov::element::f16, shape);
  memcpy(weights.data(), data, new_size);
  free(data);
  return weights;
}

void set_value_from_gguf(
    gguf_ctx* ctx,
    uint32_t type,
    gguf_value* val,
    GGUFMetaData& value) {
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
      value = ov::Tensor(ov::element::u8, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u8>::value_type>()) = val->uint8;
      break;
    case GGUF_VALUE_TYPE_INT8:
      value = ov::Tensor(ov::element::i8, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i8>::value_type>()) = val->int8;
      break;
    case GGUF_VALUE_TYPE_UINT16:
      value = ov::Tensor(ov::element::u16, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u16>::value_type>()) = val->uint16;
      break;
    case GGUF_VALUE_TYPE_INT16:
      value = ov::Tensor(ov::element::i16, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i16>::value_type>()) = val->int16;
      break;
    case GGUF_VALUE_TYPE_UINT32:
      value = ov::Tensor(ov::element::u32, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u32>::value_type>()) = val->uint32;
      break;
    case GGUF_VALUE_TYPE_INT32:
      value = ov::Tensor(ov::element::i32, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i32>::value_type>()) = val->int32;
      break;
    case GGUF_VALUE_TYPE_UINT64:
      value = ov::Tensor(ov::element::u64, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::u64>::value_type>()) = val->uint64;
      break;
    case GGUF_VALUE_TYPE_INT64:
      value = ov::Tensor(ov::element::i64, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::i64>::value_type>()) = val->int64;
      break;
    case GGUF_VALUE_TYPE_FLOAT32:
      value = ov::Tensor(ov::element::f32, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::f32>::value_type>()) = val->float32;
      break;
    case GGUF_VALUE_TYPE_BOOL:
      value = ov::Tensor(ov::element::boolean, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::boolean>::value_type>()) = val->boolval;
      break;
    case GGUF_VALUE_TYPE_STRING:
      value =
          std::string(val->string.string, static_cast<int>(val->string.len));
      break;
    case GGUF_VALUE_TYPE_FLOAT64:
      value = ov::Tensor(ov::element::f64, ov::Shape(0));
      *(std::get<ov::Tensor>(value).data<ov::element_type_traits<ov::element::f64>::value_type>()) = val->float64;
      break;
    case GGUF_VALUE_TYPE_ARRAY: {
      ctx->off += gguf_array_header_size; // Skip header
      char* data = reinterpret_cast<char*>(val) + gguf_array_header_size;
      auto size = static_cast<int>(val->array.len);
      if (val->array.type == GGUF_VALUE_TYPE_ARRAY) {
        throw std::invalid_argument(
            "[load_gguf] Only supports loading 1-layer of nested arrays.");
      }
      switch (val->array.type) {
        case GGUF_VALUE_TYPE_UINT8:
          value = ov::Tensor(ov::element::u8, {size}, reinterpret_cast<uint8_t*>(data));
          break;
        case GGUF_VALUE_TYPE_INT8:
          value = ov::Tensor(ov::element::i8, {size}, reinterpret_cast<uint8_t*>(data));
          break;
        case GGUF_VALUE_TYPE_UINT16:
          value = ov::Tensor(ov::element::u16, {size}, reinterpret_cast<uint16_t*>(data));
          break;
        case GGUF_VALUE_TYPE_INT16:
          value = ov::Tensor(ov::element::i16, {size}, reinterpret_cast<int16_t*>(data));
          break;
        case GGUF_VALUE_TYPE_UINT32:
          value = ov::Tensor(ov::element::u32, {size}, reinterpret_cast<uint32_t*>(data));
          break;
        case GGUF_VALUE_TYPE_INT32:
          value = ov::Tensor(ov::element::i32, {size}, reinterpret_cast<int32_t*>(data));
          break;
        case GGUF_VALUE_TYPE_UINT64:
          value = ov::Tensor(ov::element::u64, {size}, reinterpret_cast<uint64_t*>(data));
          break;
        case GGUF_VALUE_TYPE_INT64:
          value = ov::Tensor(ov::element::i64, {size}, reinterpret_cast<int64_t*>(data));
          break;
        case GGUF_VALUE_TYPE_FLOAT32:
          value = ov::Tensor(ov::element::f32, {size}, reinterpret_cast<float*>(data));
          break;
        case GGUF_VALUE_TYPE_BOOL:
          value = ov::Tensor(ov::element::boolean, {size}, reinterpret_cast<bool*>(data));
          break;
        case GGUF_VALUE_TYPE_STRING: {
          std::vector<std::string> strs(size);
          for (auto& str : strs) {
            auto str_val = reinterpret_cast<gguf_string*>(data);
            data += (str_val->len + sizeof(gguf_string));
            str = std::string(str_val->string, static_cast<int>(str_val->len));
            ctx->off += (str_val->len + sizeof(gguf_string));
          }
          value = std::move(strs);
          break;
        }
        case GGUF_VALUE_TYPE_FLOAT64:
          value = ov::Tensor(ov::element::f64, {size}, reinterpret_cast<double*>(data));
          break;
        default:
          throw std::runtime_error(
              "[load_gguf] Multiple levels of nested arrays are not supported.");
      }
      break;
    }
    default:
      throw std::runtime_error("[load_gguf] Received unexpected type.");
      break;
  }
  if (type == GGUF_VALUE_TYPE_STRING) {
    ctx->off += (sizeof(gguf_string) + std::get<std::string>(value).size());
  } else if (auto pv = std::get_if<ov::Tensor>(&value); pv) {
    ctx->off += pv->get_byte_size();
  }
}

std::unordered_map<std::string, GGUFMetaData> load_metadata(gguf_ctx* ctx) {
  std::unordered_map<std::string, GGUFMetaData> metadata;
  gguf_key key;
  while (gguf_get_key(ctx, &key)) {
    std::string key_name = std::string(key.name, key.namelen);
    auto& val = metadata.insert({key_name, GGUFMetaData{}}).first->second;
    set_value_from_gguf(ctx, key.type, key.val, val);
  }
  return metadata;
}

std::unordered_map<std::string, ov::Tensor> load_arrays(gguf_ctx* ctx) {
  std::unordered_map<std::string, ov::Tensor> array_map;
  gguf_tensor tensor;

  auto check_insert = [](const auto& inserted) {
    if (!inserted.second) {
      std::ostringstream msg;
      msg << "[load_gguf] Duplicate parameter name " << inserted.first->first
          << " this can happend when loading quantized tensors.";
      throw std::runtime_error(msg.str());
    }
  };

  while (gguf_get_tensor(ctx, &tensor)) {
    if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1 ||
        tensor.type == GGUF_TYPE_Q8_0) {
      gguf_load_quantized(array_map, tensor);
    } else {
      std::string name(tensor.name, tensor.namelen);
      ov::Tensor loaded_array = extract_tensor_data(&tensor); 
      check_insert(array_map.insert({name, loaded_array}));
    }
  }
  return array_map;
}

GGUFLoad get_gguf_data(const std::string& file) {
  bool exists;
  {
    std::ifstream f(file.c_str());
    exists = f.good();
  }
  if (!exists) {
    throw std::invalid_argument("[load_gguf] Failed to open " + file);
  }

  std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(
      gguf_open(file.data()), gguf_close);
  if (!ctx) {
    throw std::runtime_error("[load_gguf] gguf_init failed");
  }
  auto metadata = load_metadata(ctx.get());
  auto arrays = load_arrays(ctx.get());
  return {metadata, arrays};
}

QType get_quantization_type(int gguf_type) {
    switch(gguf_type) {
        case 0:
        case 1:
            std::cout << "Working with FP16 model" << std::endl;
            return QType::FP16;
            
        case 2:
        case 3:
            std::cout << "Working with INT4 quantized model" << std::endl;
            return QType::INT4;
            
        case 7:
            std::cout << "Working with INT8 quantized model" << std::endl;
            return QType::INT8;
            
        default:
            throw std::invalid_argument(
                "Unsupported GGUF quantization type: " + std::to_string(gguf_type));
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
    config["head_size"] = metadata_to_int(metadata, arch + ".embedding_length") / 
                     metadata_to_int(metadata, arch + ".attention.head_count");
    config["head_num_kv"] = metadata.count(arch + ".attention.head_count_kv") ?
            metadata_to_int(metadata, arch + ".attention.head_count_kv") :
            metadata_to_int(metadata, arch + ".attention.head_count");
    config["hidden_size"] = metadata_to_int(metadata, arch + ".embedding_length");
    config["max_position_embeddings"] = metadata.count(arch + ".context_length") ?
            metadata_to_int(metadata, arch + ".context_length") : 2048;
    config["rms_norm_eps"] = metadata_to_float(metadata, arch + ".attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] = metadata.count(arch + ".rope.freq_base") ?
            metadata_to_float(metadata, arch + ".rope.freq_base") : 10000.0f;
    config["qtype"] = (int)get_quantization_type(metadata_to_int(metadata, "general.file_type"));
    return config;
}

std::unordered_map<std::string, ov::Tensor> consts_from_weights(const std::map<std::string, GGUFMetaData>& config,
                                                            const std::unordered_map<std::string, ov::Tensor>& weights) {
    std::unordered_map<std::string, ov::Tensor> consts;

    consts["model.embed_tokens.weight"] = weights.at("token_embd.weight");
    consts["model.norm.weight"] = weights.at("output_norm.weight");
    if (weights.count("output.weight")) {
        consts["lm_head.weight"] = weights.at("output.weight");
        if (weights.count("output.bias")) {
          consts["lm_head.bias"] = weights.at("output.bias");
        }
    }

    // Handle quantization scales and biases
    if (weights.count("token_embd.scales")) {
        consts["model.embed_tokens.scales"] = weights.at("token_embd.scales");
        consts["model.embed_tokens.biases"] = weights.at("token_embd.biases");
    }
    if (weights.count("output.scales")) {
        consts["lm_head.scales"] = weights.at("output.scales");
        consts["lm_head.biases"] = weights.at("output.biases");
    }

    //for (auto kv : weights) std::cout << "Key: " << kv.first << std::endl;
    // Process layer weights
    for (int i = 0; i < std::get<int>(config.at("layer_num")); ++i) {
        std::string key = format("blk.%d.attn_norm.weight", i);
        consts[format("model.layers[%d].input_layernorm.weight", i)] = weights.at(format("blk.%d.attn_norm.weight", i));
        consts[format("model.layers[%d].post_attention_layernorm.weight", i)] = weights.at(format("blk.%d.ffn_norm.weight", i));
        
        // Attention weights
        consts[format("model.layers[%d].self_attn.q_proj.weight", i)] = weights.at(format("blk.%d.attn_q.weight", i));
        if (weights.count(format("blk.%d.attn_q.bias", i))) {
          consts[format("model.layers[%d].self_attn.q_proj.bias", i)] = weights.at(format("blk.%d.attn_q.bias", i));
        }
        consts[format("model.layers[%d].self_attn.k_proj.weight", i)] = weights.at(format("blk.%d.attn_k.weight", i));
        if (weights.count(format("blk.%d.attn_k.bias", i))) {
          consts[format("model.layers[%d].self_attn.k_proj.bias", i)] = weights.at(format("blk.%d.attn_k.bias", i));
        }
        consts[format("model.layers[%d].self_attn.v_proj.weight", i)] = weights.at(format("blk.%d.attn_v.weight", i));
        if (weights.count(format("blk.%d.attn_v.bias", i))) {
          consts[format("model.layers[%d].self_attn.v_proj.bias", i)] = weights.at(format("blk.%d.attn_v.bias", i));
        }
        consts[format("model.layers[%d].self_attn.o_proj.weight", i)] = weights.at(format("blk.%d.attn_output.weight", i));
        if (weights.count(format("blk.%d.attn_output.bias", i))) {
          consts[format("model.layers[%d].self_attn.o_proj.bias", i)] = weights.at(format("blk.%d.attn_output.bias", i));
        }

        // MLP weights
        consts[format("model.layers[%d].mlp.gate_proj.weight", i)] = weights.at(format("blk.%d.ffn_gate.weight", i));
        if (weights.count(format("blk.%d.ffn_gate.bias", i))) {
          consts[format("model.layers[%d].mlp.gate_proj.bias", i)] = weights.at(format("blk.%d.ffn_gate.bias", i));
        }
        consts[format("model.layers[%d].mlp.up_proj.weight", i)] = weights.at(format("blk.%d.ffn_up.weight", i));
        if (weights.count(format("blk.%d.ffn_up.bias", i))) {
          consts[format("model.layers[%d].mlp.up_proj.bias", i)] = weights.at(format("blk.%d.ffn_up.bias", i));
        }
        consts[format("model.layers[%d].mlp.down_proj.weight", i)] = weights.at(format("blk.%d.ffn_down.weight", i));
        if (weights.count(format("blk.%d.ffn_down.bias", i))) {
          consts[format("model.layers[%d].mlp.down_proj.bias", i)] = weights.at(format("blk.%d.ffn_down.bias", i));
        }

        // Quantization parameters
        if (QType(std::get<int>(config.at("qtype"))) != QType::FP16) {
            consts[format("model.layers[%d].self_attn.q_proj.scales", i)] = weights.at(format("blk.%d.attn_q.scales", i));
            consts[format("model.layers[%d].self_attn.k_proj.scales", i)] = weights.at(format("blk.%d.attn_k.scales", i));
            consts[format("model.layers[%d].self_attn.v_proj.scales", i)] = weights.at(format("blk.%d.attn_v.scales", i));
            consts[format("model.layers[%d].self_attn.o_proj.scales", i)] = weights.at(format("blk.%d.attn_output.scales", i));
            consts[format("model.layers[%d].mlp.gate_proj.scales", i)] = weights.at(format("blk.%d.ffn_gate.scales", i));
            consts[format("model.layers[%d].mlp.up_proj.scales", i)] = weights.at(format("blk.%d.ffn_up.scales", i));
            consts[format("model.layers[%d].mlp.down_proj.scales", i)] = weights.at(format("blk.%d.ffn_down.scales", i));

            consts[format("model.layers[%d].self_attn.q_proj.biases", i)] = weights.at(format("blk.%d.attn_q.biases", i));
            consts[format("model.layers[%d].self_attn.k_proj.biases", i)] = weights.at(format("blk.%d.attn_k.biases", i));
            consts[format("model.layers[%d].self_attn.v_proj.biases", i)] = weights.at(format("blk.%d.attn_v.biases", i));
            consts[format("model.layers[%d].self_attn.o_proj.biases", i)] = weights.at(format("blk.%d.attn_output.biases", i));
            consts[format("model.layers[%d].mlp.gate_proj.biases", i)] = weights.at(format("blk.%d.ffn_gate.biases", i));
            consts[format("model.layers[%d].mlp.up_proj.biases", i)] = weights.at(format("blk.%d.ffn_up.biases", i));
            consts[format("model.layers[%d].mlp.down_proj.biases", i)] = weights.at(format("blk.%d.ffn_down.biases", i));
        }
    }

    return consts;
}

std::pair<std::map<std::string, GGUFMetaData>, std::unordered_map<std::string, ov::Tensor>> load_gguf(const std::string& file) {
    auto [metadata, weights] = get_gguf_data(file);

    auto config = config_from_meta(metadata);
    auto consts = consts_from_weights(config, weights);

    return {config, consts};
}
