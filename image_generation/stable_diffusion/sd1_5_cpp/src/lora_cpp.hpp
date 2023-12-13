// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for pure C/C++ lora enabling
 * @file lora_cpp.hpp
 */

#include <Eigen/Dense>
#include <algorithm>
#include <any>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#define SAFETENSORS_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openvino/openvino.hpp>

#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "safetensors.h"

#define die(...)             \
    do {                     \
        printf(__VA_ARGS__); \
        fputc('\n', stdout); \
        exit(EXIT_FAILURE);  \
    } while (0);
#define SAFE_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SAFE_MAX(a, b) ((a) > (b) ? (a) : (b))

class InsertLoRA : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertLoRA", "0");
    std::map<std::string, std::vector<float>>* local_lora_map;
    explicit InsertLoRA(std::map<std::string, std::vector<float>>& lora_map) {
        local_lora_map = &lora_map;
        auto label = ov::pass::pattern::wrap_type<ov::op::v0::Convert>();
        // auto label = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
            // auto root = std::dynamic_pointer_cast<ov::op::v0::Constant>(m.get_match_root());
            if (!root) {
                return false;
            }
            ov::Output<ov::Node> root_output = m.get_match_value();
            std::string root_name = root->get_friendly_name();
            std::replace(root_name.begin(), root_name.end(), '.', '_');
            std::map<std::string, std::vector<float>>::iterator it = local_lora_map->begin();
            while (it != local_lora_map->end()) {
                if ((root_name).find(it->first) != std::string::npos) {
                    // std::cout << root_name << std::endl;
                    std::set<ov::Input<ov::Node>> consumers = root_output.get_target_inputs();
                    std::shared_ptr<ov::Node> lora_const =
                        ov::op::v0::Constant::create(ov::element::f32,
                                                     ov::Shape{root->get_output_shape(0)},
                                                     it->second);
                    // std::cout << "lora_const:" << lora_const->get_output_shape(0) << std::endl;
                    auto lora_add = std::make_shared<ov::opset11::Add>(root, lora_const);
                    for (auto consumer : consumers) {
                        consumer.replace_source_output(lora_add->output(0));
                    }
                    register_new_node(lora_add);
                    it = local_lora_map->erase(it);
                } else {
                    it++;
                }
            }
            return true;
        };
        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertLoRA");
        // Register Matcher
        register_matcher(m, callback);
    }
};

void debug_print_str(safetensors_Str s) {
    for (int i = 0; i < s.len; i++)
        fputc(s.ptr[i], stdout);
}

void debug_print_kv_str(safetensors_Str key, safetensors_Str value) {
    debug_print_str(key);
    printf(" = ");
    debug_print_str(value);
    fputc('\n', stdout);
}

void debug_print_kv_intlist(safetensors_Str key, IntList* intlist) {
    debug_print_str(key);
    printf(" = [");
    for (int i = 0; i < intlist->num_entries; i++)
        printf("%lli, ", (long long)intlist->entries[i]);
    printf("]\n");
}

void* read_file(const char* filename, int64_t* file_size) {
    FILE* f = fopen(filename, "rb");

    // if (!f)  die("can't open %s", filename);
    if (!f) {
        throw std::runtime_error("model init without lora\n");
    }

    if (fseek(f, 0, SEEK_END))
        die("can't fseek end on %s", filename);

#ifdef _WIN32
#    define FTELL(x) _ftelli64(x)
#else
#    define FTELL(x) ftell(x)
#endif

    int64_t pos = FTELL(f);

    if (pos == -1LL)
        die("invalid file size");
    *file_size = pos;
    if (fseek(f, 0, SEEK_SET))
        die("can't fseek start on %s", filename);

    void* buffer = malloc(*file_size);
    if (!buffer)
        die("Can't malloc %lli bytes", (long long)*file_size);

    if (*file_size != (int64_t)fread(buffer, 1, *file_size, f))
        die("cant fread");

    fclose(f);
    return buffer;
}

safetensors_Str stringToSafetensorsStr(const std::string& str) {
    safetensors_Str result;
    result.len = str.size();
    result.ptr = new char[result.len];
    memcpy(result.ptr, str.c_str(), result.len);
    return result;
}

void releaseSafetensorsStr(safetensors_Str& str) {
    delete[] str.ptr;
}

std::vector<ov::float16> readOVFloat16Data(const safetensors_TensorDescriptor& descriptor) {
    std::vector<ov::float16> data;

    if (descriptor.ptr != nullptr) {
        size_t dataSize = (descriptor.end_offset_bytes - descriptor.begin_offset_bytes) / sizeof(ov::float16);

        ov::float16* ptr = static_cast<ov::float16*>(descriptor.ptr);
        for (size_t i = 0; i < dataSize; ++i) {
            data.push_back(ov::float16(ptr[i]));
        }
    }

    return data;
}

void print_origin_layer(const std::string& key) {
    const char* filename = key.c_str();
    if (!filename) {
        std::cout << "safetensor path: " << filename << "not exist\n";
        exit(0);
    };

    int64_t sz = 0;
    void* file = read_file(filename, &sz);

    safetensors_File f = {0};

    char* result = safetensors_file_init(file, sz, &f);
    if (result) {
        std::cout << result << std::endl;

        for (char* s = SAFE_MAX(static_cast<char*>(file), f.error_context - 20);
             s < SAFE_MIN(f.error_context + 21, f.one_byte_past_end_of_header);
             s++) {
            std::cout.put(*s);
        }

        std::cout << "\n";
        std::cout << "         ^ HERE" << std::endl;
    }

    for (int i = 0; i < f.num_tensors; i++) {
        safetensors_TensorDescriptor t = f.tensors[i];
        debug_print_str(t.name);
        std::cout << "\n\tshape: (" << t.n_dimensions << ") [";

        for (int j = 0; j < t.n_dimensions; j++) {
            const char* delim = j == t.n_dimensions - 1 ? "" : ", ";
            std::cout << static_cast<long long>(t.shape[j]) << delim;
        }
        std::cout << "]\n\toffsets: [" << static_cast<long long>(t.begin_offset_bytes) << ", "
                  << static_cast<long long>(t.end_offset_bytes) << "]\n\tpointer: " << t.ptr << "\n\n";
    }
}

struct LoraWeight4Vec {
    std::vector<std::string> encoder_name;
    std::vector<std::string> unet_name;
    std::vector<std::vector<float>> encoder_value;
    std::vector<std::vector<float>> unet_value;
};

LoraWeight4Vec modify_layer(const std::string& key, float alpha = 0.75f) {
    const char* filename = key.c_str();
    // if (!filename) {
    //     std::cout << "safetensor path: " << filename << "not exist\n";
    //     exit(0);
    // };

    int64_t sz = 0;
    void* file = read_file(filename, &sz);

    safetensors_File f = {0};
    char* result = safetensors_file_init(file, sz, &f);
    if (result) {
        std::cout << result << std::endl;

        for (char* s = SAFE_MAX(static_cast<char*>(file), f.error_context - 20);
             s < SAFE_MIN(f.error_context + 21, f.one_byte_past_end_of_header);
             s++) {
            std::cout.put(*s);
        }

        std::cout << "\n";
        std::cout << "         ^ HERE" << std::endl;
    }
    // modify the layer name
    std::vector<std::map<std::string, std::string>> lora_name_list;
    // Eigen::RowMajor
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lora_weight_list;

    std::vector<std::string> visited;
    std::string LORA_PREFIX_UNET = "lora_unet";
    std::string LORA_PREFIX_TEXT_ENCODER = "lora_te";

    int counter_alpha = 0;
    int counter_w = 0;

    // loading safetensor
    for (int i = 0; i < f.num_tensors; i++) {
        std::map<std::string, std::string> lora_map;

        safetensors_TensorDescriptor tensor = f.tensors[i];
        safetensors_Str key_st_str = tensor.name;
        // char2str
        std::string key(key_st_str.ptr, key_st_str.ptr + key_st_str.len);
        // printf("origin:");
        // std::cout << key;
        // printf("\n");

        // update name of weights
        bool keyExists = std::find(visited.begin(), visited.end(), key) != visited.end();
        if (key.find(".alpha") != std::string::npos) {
            counter_alpha++;
        }

        if (key.find(".alpha") != std::string::npos || keyExists) {
            // printf("\n");
            continue;
        }

        if (key.find("text") != std::string::npos) {
            // cut out "lora_te_"
            std::string layer_infos =
                key.substr(key.find(LORA_PREFIX_TEXT_ENCODER) + LORA_PREFIX_TEXT_ENCODER.length() + 1);
            // std::cout << "layer_infos: " << layer_infos << std::endl;
            // get layer name before "."
            std::string layer_name_str = layer_infos.substr(0, layer_infos.find("."));
            // std::cout << "layer_name: " << layer_name_str << std::endl;
            // tensor.name = stringToSafetensorsStr(layer_name_str);

            // Create C++ lora_map instead of Python lora_dict
            lora_map["name"] = layer_name_str;
            lora_map["type"] = "text_encoder";

        } else {
            std::string layer_infos = key.substr(key.find(LORA_PREFIX_UNET) + LORA_PREFIX_UNET.length() + 1);
            // std::cout << "layer_infos: " << layer_infos << std::endl;
            // get layer name before "."
            std::string layer_name_str = layer_infos.substr(0, layer_infos.find("."));
            // std::cout << "layer_name: " << layer_name_str << std::endl;

            // lora_dict.update(type="unet")
            lora_map["name"] = layer_name_str;
            lora_map["type"] = "unet";
        }

        // update value of weights
        std::vector<safetensors_TensorDescriptor> pair_tensor;

        // up at first, down at second
        if (key.find("lora_down") != std::string::npos) {
            pair_tensor.push_back(f.tensors[i + 1]);
            pair_tensor.push_back(f.tensors[i]);
        } else {
            pair_tensor.push_back(f.tensors[i]);
            pair_tensor.push_back(f.tensors[i + 1]);
        }

        for (auto p_t : pair_tensor) {
            safetensors_Str key_st = p_t.name;
            std::string k_s(key_st.ptr, key_st.ptr + key_st.len);
            visited.push_back(k_s);
        }

        std::vector<int64_t> shapeVector(tensor.shape, tensor.shape + tensor.n_dimensions);

        std::vector<int64_t> shape_vec_0(pair_tensor[0].shape, pair_tensor[0].shape + pair_tensor[0].n_dimensions);
        std::vector<int64_t> shape_vec_1(pair_tensor[1].shape, pair_tensor[1].shape + pair_tensor[1].n_dimensions);

        // read float16 weight
        auto raw_data_0 = readOVFloat16Data(pair_tensor[0]);
        auto raw_data_1 = readOVFloat16Data(pair_tensor[1]);

        // matmul with fp16
        Eigen::Map<Eigen::Matrix<ov::float16, Eigen::Dynamic, Eigen::Dynamic>> mat2d_f16_0(raw_data_0.data(),
                                                                                           shape_vec_0[0],
                                                                                           shape_vec_0[1]);
        Eigen::Map<Eigen::Matrix<ov::float16, Eigen::Dynamic, Eigen::Dynamic>> mat2d_f16_1(raw_data_1.data(),
                                                                                           shape_vec_1[0],
                                                                                           shape_vec_1[1]);

        // matmul with floats
        std::vector<float> float_data_0(raw_data_0.begin(), raw_data_0.end());
        std::vector<float> float_data_1(raw_data_1.begin(), raw_data_1.end());
        int64_t rows_0 = shape_vec_0[0];
        int64_t cols_0 = shape_vec_0[1];
        int64_t rows_1 = shape_vec_1[0];
        int64_t cols_1 = shape_vec_1[1];
        // RowMajor
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat2d_f_0(float_data_0.data(),
                                                                                                    rows_0,
                                                                                                    cols_0);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat2d_f_1(float_data_1.data(),
                                                                                                    rows_1,
                                                                                                    cols_1);

        auto start_time_f = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matmul_f = alpha * mat2d_f_0 * mat2d_f_1;
        auto end_time_f = std::chrono::steady_clock::now();
        auto duration_f = std::chrono::duration_cast<std::chrono::duration<float>>(end_time_f - start_time_f);
        // std::cout << "Matrix multiplication time: " << duration_f.count() << " seconds" << std::endl;

        counter_w++;

        lora_name_list.push_back(lora_map);
        lora_weight_list.push_back(matmul_f);
    }
    // std::cout << "\n End of safetensor modification, next step: convert mat to vector, after matmul \n" ;

    std::vector<std::string> encoder_layers;
    std::vector<std::string> unet_layers;
    std::vector<std::vector<float>> encoder_vec;
    std::vector<std::vector<float>> unet_vec;

    // from eigen::matrix to vector, after matmul
    for (size_t i = 0; i < lora_name_list.size(); i++) {
        std::map<std::string, std::string> lora_map = lora_name_list[i];
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weight_mat = lora_weight_list[i];

        std::string name = lora_map["name"];
        std::string type = lora_map["type"];

        // std::cout << "Name: " << name << ", Type: " << type << ", Shape (" << weight_mat.rows() << ", " <<
        // weight_mat.cols() << ")" << std::endl;

        if (lora_map["type"] == "text_encoder") {
            // get_encoder_layers
            encoder_layers.push_back(lora_map["name"]);

            std::vector<float> float_data(weight_mat.size());
            Eigen::Map<Eigen::VectorXf> float_vec(float_data.data(), float_data.size());
            float_vec = Eigen::Map<Eigen::VectorXf>(weight_mat.data(), weight_mat.size());
            encoder_vec.push_back(float_data);
        } else {
            // get_unet_layers
            unet_layers.push_back(lora_map["name"]);

            std::vector<float> float_data(weight_mat.size());
            Eigen::Map<Eigen::VectorXf> float_vec(float_data.data(), float_data.size());
            float_vec = Eigen::Map<Eigen::VectorXf>(weight_mat.data(), weight_mat.size());
            unet_vec.push_back(float_data);
        }
    }
    // std::cout << "encoder_vec.size: " << encoder_vec.size() << " , unet_vec.size: " << unet_vec.size() << std::endl;
    // std::cout << "\n End of implementation lora.hpp after loading safetensor \n" ;

    LoraWeight4Vec results;
    results.encoder_name = encoder_layers;
    results.unet_name = unet_layers;
    results.encoder_value = encoder_vec;
    results.unet_value = unet_vec;

    return results;
}

std::vector<ov::CompiledModel> load_lora_weights_cpp(ov::Core& core,
                                                     std::shared_ptr<ov::Model>& text_encoder_model,
                                                     std::shared_ptr<ov::Model>& unet_model,
                                                     const std::string& device,
                                                     const std::map<std::string, float>& lora_models) {
    std::vector<ov::CompiledModel> compiled_lora_models;

    if (!lora_models.empty()) {
        std::map<std::string, std::vector<float>> lora_map;
        ov::pass::Manager manager;
        int flag = 0;
        try {
            auto start = std::chrono::steady_clock::now();
            for (std::map<std::string, float>::const_iterator it = lora_models.begin(); it != lora_models.end(); ++it) {
                // DEBUG:
                // print_origin_layer(it->first);

                // divide into 2 groups with name
                // use Eigen to implement matmul
                LoraWeight4Vec new_layers = modify_layer(it->first, it->second);
                std::vector<std::vector<float>> encoder_value_vec = new_layers.encoder_value;
                std::vector<std::vector<float>> unet_value_vec = new_layers.unet_value;

                //-------------get text encoder vectors-----------
                flag = 0;
                for (auto item : encoder_value_vec) {
                    lora_map.insert(std::pair<std::string, std::vector<float>>(new_layers.encoder_name[flag], item));
                    flag++;
                }
                //-------------get unet vectors-------------------
                flag = 0;
                for (auto item : unet_value_vec) {
                    lora_map.insert(std::pair<std::string, std::vector<float>>(new_layers.unet_name[flag], item));
                    flag++;
                }
                auto end = std::chrono::steady_clock::now();
                std::cout << "lora_extract:" << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
                          << std::endl;
            }
            manager.register_pass<InsertLoRA>(lora_map);
            auto start_txt = std::chrono::steady_clock::now();
            // if(!encoder_layers.empty()){
            if (!lora_models.empty()) {
                manager.run_passes(text_encoder_model);
            }
            auto end_txt = std::chrono::steady_clock::now();
            compiled_lora_models.push_back(core.compile_model(text_encoder_model, device));
            auto start_unet = std::chrono::steady_clock::now();
            manager.run_passes(unet_model);
            auto end_unet = std::chrono::steady_clock::now();
            compiled_lora_models.push_back(core.compile_model(unet_model, device));

            std::cout << "text_encoder run pass:"
                      << std::chrono::duration<double, std::milli>(end_txt - start_txt).count() << " ms" << std::endl;
            std::cout << "unet run pass:" << std::chrono::duration<double, std::milli>(end_unet - start_unet).count()
                      << " ms" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << e.what();
            compiled_lora_models.push_back(core.compile_model(text_encoder_model, device));
            compiled_lora_models.push_back(core.compile_model(unet_model, device));
            return compiled_lora_models;
        }
    }
    return compiled_lora_models;
}