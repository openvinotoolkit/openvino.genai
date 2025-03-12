#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>

#include "gguf.h"

// https://github.com/antirez/gguf-tools/blob/af7d88d808a7608a33723fba067036202910acb3/gguflib.h#L102-L108
constexpr int gguf_array_header_size = 12;

using GGUFLoad = std::pair<
    std::unordered_map<std::string, GGUFMetaData>,
    std::unordered_map<std::string, ov::Tensor>>;

std::string format(const std::string fmt_str, ...) {
    va_list ap;
    char *fp = NULL;
    va_start(ap, fmt_str);
    vasprintf(&fp, fmt_str.c_str(), ap);
    va_end(ap);
    std::unique_ptr<char[]> formatted(fp);
    return std::string(formatted.get());
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
      return {};
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
      return {};
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

void append_kv_array(
    gguf_ctx* ctx,
    const std::string& key,
    ov::Tensor& val,
    uint32_t gguf_type) {
  if (val.get_shape().size() == 1) {
    size_t gguf_size = val.get_byte_size() + gguf_array_header_size;
    std::vector<char> val_vec(gguf_size);
    gguf_value* gguf_val = reinterpret_cast<gguf_value*>(val_vec.data());
    gguf_val->array.type = gguf_type;
    gguf_val->array.len = val.get_size();
    memcpy(
        val_vec.data() + gguf_array_header_size,
        val.data<ov::element_type_traits<ov::element::u8>::value_type>(),
        val.get_byte_size());
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        GGUF_VALUE_TYPE_ARRAY,
        reinterpret_cast<void*>(val_vec.data()),
        gguf_size);
  } else {
    gguf_append_kv(
        ctx,
        key.c_str(),
        key.length(),
        gguf_type,
        reinterpret_cast<void*>(val.data<ov::element_type_traits<ov::element::u8>::value_type>()),
        val.get_byte_size());
  }
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
    config["layer_num"] = metadata_to_int(metadata, "llama.block_count");
    config["head_num"] = metadata_to_int(metadata, "llama.attention.head_count");
    config["head_size"] = metadata_to_int(metadata, "llama.embedding_length") / 
                     metadata_to_int(metadata, "llama.attention.head_count");
    config["head_num_kv"] = metadata.count("llama.attention.head_count_kv") ?
            metadata_to_int(metadata, "llama.attention.head_count_kv") :
            metadata_to_int(metadata, "llama.attention.head_count");
    config["hidden_size"] = metadata_to_int(metadata, "llama.embedding_length");
    config["max_position_embeddings"] = metadata.count("llama.context_length") ?
            metadata_to_int(metadata, "llama.context_length") : 2048;
    config["rotary_dims"] = metadata_to_int(metadata, "llama.rope.dimension_count");
    config["rms_norm_eps"] = metadata_to_float(metadata, "llama.attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] = metadata.count("llama.rope.freq_base") ?
            metadata_to_float(metadata, "llama.rope.freq_base") : 10000.0f;
    config["qtype"] = (int)get_quantization_type(metadata_to_int(metadata, "general.file_type"));

    config["architecture"] = std::get<std::string>(metadata.at("general.architecture"));

    return config;
}

std::unordered_map<std::string, ov::Tensor> consts_from_weights(const std::map<std::string, GGUFMetaData>& config,
                                                            const std::unordered_map<std::string, ov::Tensor>& weights) {
    std::unordered_map<std::string, ov::Tensor> consts;

    consts["model.embed_tokens.weight"] = weights.at("token_embd.weight");
    consts["model.norm.weight"] = weights.at("output_norm.weight");
    consts["lm_head.weight"] = weights.count("output.weight") ? 
        weights.at("output.weight") : ov::Tensor();

    // Handle quantization scales and biases
    if (weights.count("token_embd.scales")) {
        consts["model.embed_tokens.scales"] = weights.at("token_embd.scales");
        consts["model.embed_tokens.biases"] = weights.at("token_embd.biases");
    }
    if (weights.count("output.scales")) {
        consts["lm_head.scales"] = weights.at("output.scales");
        consts["lm_head.biases"] = weights.at("output.biases");
    }

    // Process layer weights
    for (int i = 0; i < std::get<int>(config.at("layer_num")); ++i) {
        consts[format("model.layers[{}].input_layernorm.weight", i)] = weights.at(format("blk.{}.attn_norm.weight", i));
        consts[format("model.layers[{}].post_attention_layernorm.weight", i)] = weights.at(format("blk.{}.ffn_norm.weight", i));
        
        // Attention weights
        consts[format("model.layers[{}].self_attn.q_proj.weight", i)] = weights.at(format("blk.{}.attn_q.weight", i));
        consts[format("model.layers[{}].self_attn.k_proj.weight", i)] = weights.at(format("blk.{}.attn_k.weight", i));
        consts[format("model.layers[{}].self_attn.v_proj.weight", i)] = weights.at(format("blk.{}.attn_v.weight", i));
        consts[format("model.layers[{}].self_attn.o_proj.weight", i)] = weights.at(format("blk.{}.attn_output.weight", i));

        // MLP weights
        consts[format("model.layers[{}].mlp.gate_proj.weight", i)] = weights.at(format("blk.{}.ffn_gate.weight", i));
        consts[format("model.layers[{}].mlp.up_proj.weight", i)] = weights.at(format("blk.{}.ffn_up.weight", i));
        consts[format("model.layers[{}].mlp.down_proj.weight", i)] = weights.at(format("blk.{}.ffn_down.weight", i));

        // Quantization parameters
        if (QType(std::get<int>(config.at("qtype"))) != QType::FP16) {
            consts[format("model.layers[{}].self_attn.q_proj.scales", i)] = weights.at(format("blk.{}.attn_q.scales", i));
            consts[format("model.layers[{}].self_attn.k_proj.scales", i)] = weights.at(format("blk.{}.attn_k.scales", i));
            consts[format("model.layers[{}].self_attn.v_proj.scales", i)] = weights.at(format("blk.{}.attn_v.scales", i));
            consts[format("model.layers[{}].self_attn.o_proj.scales", i)] = weights.at(format("blk.{}.attn_output.scales", i));
            consts[format("model.layers[{}].mlp.gate_proj.scales", i)] = weights.at(format("blk.{}.ffn_gate.scales", i));
            consts[format("model.layers[{}].mlp.up_proj.scales", i)] = weights.at(format("blk.{}.ffn_up.scales", i));
            consts[format("model.layers[{}].mlp.down_proj.scales", i)] = weights.at(format("blk.{}.ffn_down.scales", i));

            consts[format("model.layers[{}].self_attn.q_proj.biases", i)] = weights.at(format("blk.{}.attn_q.biases", i));
            consts[format("model.layers[{}].self_attn.k_proj.biases", i)] = weights.at(format("blk.{}.attn_k.biases", i));
            consts[format("model.layers[{}].self_attn.v_proj.biases", i)] = weights.at(format("blk.{}.attn_v.biases", i));
            consts[format("model.layers[{}].self_attn.o_proj.biases", i)] = weights.at(format("blk.{}.attn_output.biases", i));
            consts[format("model.layers[{}].mlp.gate_proj.biases", i)] = weights.at(format("blk.{}.ffn_gate.biases", i));
            consts[format("model.layers[{}].mlp.up_proj.biases", i)] = weights.at(format("blk.{}.ffn_up.biases", i));
            consts[format("model.layers[{}].mlp.down_proj.biases", i)] = weights.at(format("blk.{}.ffn_down.biases", i));
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

// void save_gguf(
//     std::string file,
//     std::unordered_map<std::string, array> array_map,
//     std::unordered_map<std::string, GGUFMetaData> metadata /* = {} */) {
//   // Add .gguf to file name if it is not there
//   if (file.length() < 5 || file.substr(file.length() - 5, 5) != ".gguf") {
//     file += ".gguf";
//   }

//   std::unique_ptr<gguf_ctx, decltype(&gguf_close)> ctx(
//       gguf_create(file.c_str(), GGUF_OVERWRITE), gguf_close);
//   if (!ctx) {
//     throw std::runtime_error("[save_gguf] gguf_create failed");
//   }

//   auto string_to_gguf = [](char* dst, const std::string& src) {
//     gguf_string* val = reinterpret_cast<gguf_string*>(dst);
//     val->len = src.length();
//     memcpy(val->string, src.c_str(), src.length());
//   };

//   // Save any meta data
//   for (auto& [key, value] : metadata) {
//     if (auto pv = std::get_if<std::string>(&value); pv) {
//       const std::string& str = *pv;
//       size_t size = sizeof(gguf_string) + str.length();
//       std::vector<char> val_vec(size);
//       string_to_gguf(val_vec.data(), str);
//       gguf_append_kv(
//           ctx.get(),
//           key.c_str(),
//           key.length(),
//           GGUF_VALUE_TYPE_STRING,
//           static_cast<void*>(val_vec.data()),
//           size);
//     } else if (auto pv = std::get_if<std::vector<std::string>>(&value); pv) {
//       const auto& str_vec = *pv;
//       auto mem_size = std::accumulate(
//           str_vec.begin(), str_vec.end(), 0, [](size_t accum, const auto& s) {
//             return accum + s.size();
//           });
//       mem_size += str_vec.size() * sizeof(gguf_string) + gguf_array_header_size;
//       std::vector<char> val_vec(mem_size);
//       gguf_value* val = reinterpret_cast<gguf_value*>(val_vec.data());
//       val->array.type = GGUF_VALUE_TYPE_STRING;
//       val->array.len = str_vec.size();
//       auto str_ptr = val_vec.data() + gguf_array_header_size;
//       for (auto& str : str_vec) {
//         string_to_gguf(str_ptr, str);
//         str_ptr += str.length() + sizeof(gguf_string);
//       }
//       gguf_append_kv(
//           ctx.get(),
//           key.c_str(),
//           key.length(),
//           GGUF_VALUE_TYPE_ARRAY,
//           static_cast<void*>(val),
//           mem_size);
//     } else if (auto pv = std::get_if<array>(&value); pv) {
//       array v = *pv;
//       if (v.ndim() > 1) {
//         throw std::runtime_error(
//             "[save_gguf] Cannot save arrays with more than one dimension.");
//       }
//       if (v.size() == 0) {
//         throw std::runtime_error("[save_gguf] Cannot save empty arrays.");
//       }

//       eval(v);
//       if (!v.flags().row_contiguous) {
//         v = reshape(flatten(v), v.shape());
//       }
//       if (!v.flags().row_contiguous) {
//         throw std::runtime_error(
//             "[save_gguf] Cannot save non contiguous arrays.");
//       }
//       switch (v.dtype()) {
//         case float32:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_FLOAT32);
//           break;
//         case int64:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT64);
//           break;
//         case int32:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT32);
//           break;
//         case int16:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT16);
//           break;
//         case int8:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_INT8);
//           break;
//         case uint64:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT64);
//           break;
//         case uint32:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT32);
//           break;
//         case uint16:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT16);
//           break;
//         case uint8:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_UINT8);
//           break;
//         case bool_:
//           append_kv_array(ctx.get(), key, v, GGUF_VALUE_TYPE_BOOL);
//           break;
//         default:
//           std::ostringstream msg;
//           msg << "[save_gguf] array type " << v.dtype()
//               << " not support for metadata.";
//           throw std::invalid_argument(msg.str());
//       }
//     } else {
//       throw std::runtime_error(
//           "[save_gguf] Received unexpected type in metadata");
//     }
//   }

//   // Tensor offsets are relative to data section, so we start at offset 0.
//   uint64_t tensor_offset = 0;

//   // First, append the tensor info
//   for (auto& [key, arr] : array_map) {
//     arr.eval();

//     // Try to make it row contiguous
//     if (!arr.flags().row_contiguous) {
//       arr = reshape(flatten(arr), arr.shape());
//       arr.eval();
//     }

//     // Has to be row-major now but, check one more time in case
//     // any of the above change in the future
//     if (!arr.flags().row_contiguous) {
//       throw std::invalid_argument(
//           "[save_gguf] can only serialize row-major arrays");
//     }

//     tensor_offset += gguf_get_alignment_padding(ctx->alignment, tensor_offset);
//     const std::optional<uint32_t> gguf_type =
//         dtype_to_gguf_tensor_type(arr.dtype());
//     if (!gguf_type.has_value()) {
//       std::ostringstream msg;
//       msg << "[save_gguf] dtype " << arr.dtype() << " is not supported";
//       throw std::runtime_error(msg.str());
//     }
//     const char* tensorname = key.c_str();
//     const uint64_t namelen = key.length();
//     const uint32_t num_dim = arr.ndim();
//     uint64_t dim[num_dim];
//     for (int i = 0; i < num_dim; i++) {
//       dim[i] = arr.shape()[num_dim - 1 - i];
//     }
//     if (!gguf_append_tensor_info(
//             ctx.get(),
//             tensorname,
//             namelen,
//             num_dim,
//             dim,
//             gguf_type.value(),
//             tensor_offset)) {
//       throw std::runtime_error("[save_gguf] gguf_append_tensor_info failed");
//     }
//     tensor_offset += arr.nbytes();
//   }

//   // Then, append the tensor weights
//   for (const auto& [key, arr] : array_map) {
//     if (!gguf_append_tensor_data(
//             ctx.get(), (void*)arr.data<void>(), arr.nbytes())) {
//       throw std::runtime_error("[save_gguf] gguf_append_tensor_data failed");
//     }
//   }
// }

