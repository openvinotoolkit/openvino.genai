// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"
#include <fstream>

namespace {

// todo: remove when openvino-tokenizers will support left padding
ov::genai::TokenizedInputs pad_left(ov::Tensor& input_ids, ov::Tensor& attention_mask, int64_t pad_token_id) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token_id)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token_id)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

#ifdef _WIN32
#    include <windows.h>
#    define MAX_ABS_PATH _MAX_PATH
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
#else
#    include <dlfcn.h>
#    include <limits.h>
#    define MAX_ABS_PATH PATH_MAX
#    define get_absolute_path(result, path) realpath(path.c_str(), result)

std::string get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    std::ignore = get_absolute_path(&absolutePath[0], path);
    if (!absolutePath.empty()) {
        // on Linux if file does not exist or no access, function will return NULL, but
        // `absolutePath` will contain resolved path
        absolutePath.resize(absolutePath.find('\0'));
        return std::string(absolutePath);
    }
    std::stringstream ss;
    ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
    throw std::runtime_error(ss.str());
}
#endif

std::string get_ov_genai_library_path() {
    #ifdef _WIN32
        CHAR genai_library_path[MAX_PATH];
        HMODULE hm = NULL;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPSTR>(get_ov_genai_library_path),
                                &hm)) {
            std::stringstream ss;
            ss << "GetModuleHandle returned " << GetLastError();
            throw std::runtime_error(ss.str());
        }
        GetModuleFileNameA(hm, (LPSTR)genai_library_path, sizeof(genai_library_path));
        return std::string(genai_library_path);
    #elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
        Dl_info info;
        dladdr(reinterpret_cast<void*>(get_ov_genai_library_path), &info);
        return get_absolute_file_path(info.dli_fname).c_str();
    #else
    #    error "Unsupported OS"
    #endif  // _WIN32
}

std::filesystem::path with_openvino_tokenizers(const std::filesystem::path& path) {
    #ifdef _WIN32
        constexpr char tokenizers[] = "openvino_tokenizers.dll";
    #elif __linux__
        constexpr char tokenizers[] = "libopenvino_tokenizers.so";
    #elif __APPLE__
        constexpr char tokenizers[] = "libopenvino_tokenizers.dylib";
    #endif
        return path.parent_path() / tokenizers;
}

constexpr char bos_token_key_name[] = "bos_token";
constexpr char eos_token_key_name[] = "eos_token";      
constexpr char pad_token_key_name[] = "pad_token";

}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;

    std::string m_pad_token = "";
    std::string m_bos_token = "";
    std::string m_eos_token = "";

    TokenizerImpl() = default;

    TokenizerImpl(std::filesystem::path tokenizer_path) {
        ov::Core core;
        
        if (tokenizer_path.extension() == ".xml")
            OPENVINO_THROW("ov_tokenizers_path should be a path to a dir not a xml file");

        const char* ov_tokenizers_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
        if (ov_tokenizers_path) {
            core.add_extension(ov_tokenizers_path);
        } else {
            OPENVINO_THROW("openvino_tokenizers path is not set");
        }
        
        read_config(tokenizer_path);
        read_special_tokens_map(tokenizer_path);

        // Try to read tokenizer_config if some token ids or token str are not defined.
        read_tokenizer_config_if_necessary(tokenizer_path); 

        auto device = "CPU"; // currently openvino_tokenizer supports only CPU
        m_tokenize_request = core.compile_model(tokenizer_path / "openvino_tokenizer.xml", 
                                                device).create_infer_request();
        m_detokenizer_request = core.compile_model(tokenizer_path / "openvino_detokenizer.xml", 
                                                   device).create_infer_request();

        // Get special token ids by inference if they are not defined.
        // todo: do not call until CVS-143410 is resolved
        // infer_special_tokens_if_necessary();
    }

    // load special tokens ids from config.json
    void read_config(const std::filesystem::path& tokenizer_path) {
        auto config_file_path = tokenizer_path / "config.json";
        if (!std::filesystem::exists(config_file_path))
            return ;
        std::ifstream file(config_file_path);
        if (!file.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(file);
        using ov::genai::utils::read_json_param;

        read_json_param(data, "pad_token_id", m_pad_token_id);
        read_json_param(data, "bos_token_id", m_bos_token_id);
        read_json_param(data, "eos_token_id", m_eos_token_id);
    }

    // Reads the string representation of special tokens if they exist.
    void read_special_tokens_map(const std::filesystem::path& tokenizer_path) {
        auto special_tokens_file_path = tokenizer_path / "special_tokens_map.json";
        if (!std::filesystem::exists(special_tokens_file_path))
            return ;
        std::ifstream f(special_tokens_file_path);
        if (!f.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(f);

        using ov::genai::utils::read_json_param;
        // they are in the format {"bos_token": { "content": "<s>",... }}
        auto read_token_content_str = [&data](std::string key_name, std::string& val) {
            if (val == "" && data.contains(key_name)) { read_json_param(data[key_name], "content", val); }
        };
        read_token_content_str(pad_token_key_name, m_pad_token);
        read_token_content_str(bos_token_key_name, m_bos_token);
        read_token_content_str(eos_token_key_name, m_eos_token);
    }

    // Read string representation of special tokens if they exists.
    // Also tries to load special token ids from added_tokens_decoder if they exist.
    // Will not override special token strings or ids if they already exist
    void read_tokenizer_config_if_necessary(const std::filesystem::path& tokenizer_path) {
        if (m_pad_token_id != -1 && m_bos_token_id != -1 && m_eos_token_id != -1 && 
            !m_pad_token.empty() && !m_bos_token.empty() && !m_eos_token.empty()) {
            return ;
        }

        auto tokenizer_config_file_path = tokenizer_path / "tokenizer_config.json";
        if (!std::filesystem::exists(tokenizer_config_file_path))
            return ;
        std::ifstream f(tokenizer_config_file_path);
        if (!f.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(f);

        // read special tokens string representation 
        // if they are presented directly {"bos_token": "<bos>"}
        using ov::genai::utils::read_json_param;
        auto read_token_str = [&data](std::string key_name, std::string& val) {
            if (val.empty()) { read_json_param(data, key_name, val); }
        };
        read_token_str(pad_token_key_name, m_pad_token);
        read_token_str(bos_token_key_name, m_bos_token);
        read_token_str(eos_token_key_name, m_eos_token);

        // if special tokens are not loaded directly, try to read
        // if they are in the format {"bos_token": { "content": "<s>",... }}
        auto read_token_content_str = [&data](std::string key_name, std::string& val) {
            if (val.empty() && data.contains(key_name)) { read_json_param(data[key_name], "content", val); }
        };
        read_token_content_str(pad_token_key_name, m_pad_token);
        read_token_content_str(bos_token_key_name, m_bos_token);
        read_token_content_str(eos_token_key_name, m_eos_token);

        // special token ids integer representation are already defined
        if (m_pad_token_id != -1 && m_bos_token_id != -1 && m_eos_token_id != -1)
            return ;

        // values are stored as {"added_tokens_decoder": {"0": {"content": "<pad>"}}}
        // token id is a key in the form of a string, need to do std::stoi
        std::string spec_tokens_key_name = "added_tokens_decoder";
        if (!data.contains(spec_tokens_key_name))
            return ;

        // if added_tokens_decoder has different format items() will not fail
        for (auto& [key, value] : data[spec_tokens_key_name].items()) {
            if (!value.contains("content"))
                continue;
            auto content = value["content"];
            if (m_pad_token_id == -1 && content == m_pad_token)
                m_pad_token_id = std::stoi(key);
            if (m_bos_token_id == -1 && content == m_bos_token)
                m_bos_token_id = std::stoi(key);
            if (m_eos_token_id == -1 && content == m_eos_token)
                m_eos_token_id = std::stoi(key);
        }
    }

    // tokenize str representation to get special tokens integer values
    void infer_special_tokens_if_necessary() {
        auto get_id_from_str = [this](std::string token_str, int64_t& token_val) {
            if (token_val != -1 || token_str.empty()) 
                return ;
            auto token_ids_tensor = this->encode(token_str).input_ids;
            auto data = token_ids_tensor.data<int64_t>();
            auto data_len = token_ids_tensor.get_shape()[1];
            token_val = data[data_len - 1];
        };
        get_id_from_str(m_pad_token, m_pad_token_id);
        get_id_from_str(m_bos_token, m_bos_token_id);
        get_id_from_str(m_eos_token, m_eos_token_id);
    }

    TokenizedInputs encode(std::string prompt) {
        size_t batch_size = 1;
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
        m_tokenize_request.infer();
        return get_copied_results();
    }

    TokenizedInputs encode(std::vector<std::string>& prompts) {
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
        auto size_ = m_tokenize_request.get_input_tensor().get_shape();
        m_tokenize_request.infer();
       
        auto res = get_copied_results();
        pad_left(res.input_ids, res.attention_mask, m_pad_token_id);
        return {res.input_ids, res.attention_mask};
    }

    TokenizedInputs get_copied_results() {
        auto input_ids = m_tokenize_request.get_tensor("input_ids");
        auto attention_mask = m_tokenize_request.get_tensor("attention_mask");
        ov::Tensor input_ids_ = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
        ov::Tensor attention_mask_ = ov::Tensor(attention_mask.get_element_type(), attention_mask.get_shape());
        input_ids.copy_to(input_ids_);
        attention_mask.copy_to(attention_mask_);

        return {input_ids_, attention_mask_};        
    }

    std::string decode(std::vector<int64_t> tokens) {
        size_t batch_size = 1;
        m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        m_detokenizer_request.infer();
        return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens) {
        OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64, "tokens tensor element type should be an i64");
        OPENVINO_ASSERT(tokens.get_shape().size() == 2, "tokens tensor should of rank 2 with shape [batch_size, seq_len]");

        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines) {
        auto compare_lengths = [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
            return a.size() < b.size();
        };
        size_t max_len = std::max_element(lines.begin(), lines.end(), compare_lengths)->size();

        ov::Tensor tokens = ov::Tensor{ov::element::i64, {lines.size(), max_len}};
        auto tokens_data = tokens.data<int64_t>();
        
        for (size_t i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            size_t line_len = line.size();
            std::copy(line.begin(), line.end(), tokens_data + i * max_len);
            std::fill(tokens_data + i * max_len + line_len, tokens_data + (i + 1) * max_len, m_pad_token_id);
        }

        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }
};

Tokenizer::Tokenizer(const std::string& tokenizer_path) {
    ov::genai::ScopedVar env_manager(tokenizers_relative_to_genai().string());
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizer_path);
}

TokenizedInputs Tokenizer::encode(const std::string prompt) {
    return m_pimpl->encode(std::move(prompt));
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text) {
    return encode(std::vector<std::string>(text.begin(), text.end()));
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
}

int64_t Tokenizer::get_bos_token_id() const {
    return m_pimpl->m_bos_token_id;
}

int64_t Tokenizer::get_eos_token_id() const {
    return m_pimpl->m_eos_token_id;
}

int64_t Tokenizer::get_pad_token_id() const {
    return m_pimpl->m_pad_token_id;
}

std::string Tokenizer::get_pad_token() const {
    return m_pimpl->m_pad_token;
}

std::string Tokenizer::get_bos_token() const {
    return m_pimpl->m_bos_token;
}

std::string Tokenizer::get_eos_token() const {
    return m_pimpl->m_eos_token;
}

Tokenizer::~Tokenizer() = default;

std::filesystem::path tokenizers_relative_to_genai() {
    return with_openvino_tokenizers(get_ov_genai_library_path());
}

ScopedVar::ScopedVar(const std::string& environment_variable_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    _dupenv_s(&value, &len, ENVIRONMENT_VARIABLE_NAME);
    if (value == nullptr)
        _putenv_s(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str());
#else
    if (!getenv(ENVIRONMENT_VARIABLE_NAME))
        setenv(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str(), 1);
#endif
    else
        was_already_set = true;
}

ScopedVar::~ScopedVar() {
    if (!was_already_set) {
#ifdef _WIN32
        _putenv_s(ENVIRONMENT_VARIABLE_NAME, "");
#else
        unsetenv(ENVIRONMENT_VARIABLE_NAME);
#endif
    }
}
}  // namespace genai
}  // namespace ov
