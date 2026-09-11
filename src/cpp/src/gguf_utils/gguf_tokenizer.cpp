// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>

#include "gguf_tokenizer.hpp"

#include "openvino/frontend/gguf/tokenizer_metadata.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

using namespace ov::op;


constexpr int32_t MAX_LENGTH = 8192;
constexpr float VOCAB_SIZE_CACHE_PROPORTION = 0.2f;
constexpr int32_t MIN_CACHE_CAPACITY = 20'000;

namespace ov {
namespace genai {
bool is_gguf_model(const std::filesystem::path& file_path) {
    return file_path.extension() == ".gguf";
}

std::map<std::string, GGUFMetaData> tokenizer_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> tokenizer_config;

    const std::string prefix = "tokenizer.";
    for (const auto& [key, value] : metadata) {
        if (key.compare(0, prefix.size(), prefix) == 0) {
            size_t last_dot = key.find_last_of('.');
            // Extract the last part after "."
            std::string sub_key = (last_dot != std::string_view::npos) ? std::string(key.substr(last_dot + 1)) : key;
            tokenizer_config[sub_key] = value;
        }
    }

    return tokenizer_config;
}

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            FreeLibrary(static_cast<HMODULE>(h));
    });
#else
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string() + "\n" + dlerror());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            dlclose(h);
    });
#endif
}

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName) {
    if (!shared_object || !symbolName) {
        throw std::invalid_argument("Null shared object or symbol name.");
    }

#ifdef _WIN32
    HMODULE handle = static_cast<HMODULE>(shared_object.get());
    void* symbol = reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
    if (!symbol) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName));
    }
    return symbol;
#else
    void* handle = shared_object.get();
    // Clear existing errors
    dlerror();
    void* symbol = dlsym(handle, symbolName);
    const char* error = dlerror();
    if (error) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName) + "\n" + error);
    }
    return symbol;
#endif
}

ov::OutputVector add_ragged_dimension(const ov::OutputVector& inputs) {
    auto input_shape = std::make_shared<v3::ShapeOf>(inputs[0], element::i32);
    auto const_zero = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto const_one = std::make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto batch_size = std::make_shared<v8::Gather>(input_shape, const_zero, const_zero);
    auto batch_size_plus_one = std::make_shared<v1::Add>(batch_size, const_one);
    auto ragged_begins = std::make_shared<v4::Range>(const_zero, batch_size, const_one, element::i32)->output(0);
    auto ragged_ends = std::make_shared<v4::Range>(const_one, batch_size_plus_one, const_one, element::i32)->output(0);

    ov::OutputVector res = ov::OutputVector{ragged_begins, ragged_ends};
    res.insert(res.end(), inputs.begin(), inputs.end());
    return res;
}

bool is_special_token(int32_t token_type) {
    return token_type == 3 || token_type == 4;
}

std::string quote_meta(const std::string& str) {
    std::string result = "(";
    
    // todo: add also utf validate
    for (char c : str) {
        if (!std::isalnum(c) && c != '_') {
            result += '\\';
        }
        result += c;
    }
    result += ")";
    return result;
}

std::string join_special_tokens(const std::vector<std::string>& special_tokens) {
    std::ostringstream oss;
    for (size_t i = 0; i < special_tokens.size(); ++i) {
        if (i > 0)
            oss << "|";
        oss << quote_meta(special_tokens[i]);
    }
    return oss.str();
}

std::vector<std::string> get_split_regex(const std::string& pre) {
    // taken from
    // https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/src/llama-vocab.cpp#L279
    // TODO: complete for other archs
    static std::unordered_map<std::string, std::vector<std::string>> regex_map = {
        {"qwen2",
         {
             // original regex from tokenizer.json
             // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|
             // ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
             "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
         }},
        {"smollm",
         {
             "\\p{N}",
             "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
         }},
    };

    if (regex_map.count(pre)) {
        return regex_map.at(pre);
    }

    std::vector<std::string> default_regex_exprs = {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]",
    };

    return default_regex_exprs;
}

ov::OutputVector create_string_constant(const std::vector<std::string>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

ov::OutputVector create_string_constant(const std::vector<std::vector<uint8_t>>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

const std::unordered_map<std::string, uint8_t>& unicode_to_bytes() {
    static const std::unordered_map<std::string, uint8_t> map = []() {
        std::vector<uint8_t> bs;

        // Range: '!' (33) to '~' (126)
        for (uint8_t i = static_cast<uint8_t>('!'); i <= static_cast<uint8_t>('~'); ++i) {
            bs.push_back(i);
        }

        // Range: '¡' (161) to '¬' (172)
        for (uint8_t i = 0xA1; i <= 0xAC; ++i) {
            bs.push_back(i);
        }

        // Range: '®' (174) to 'ÿ' (255)
        for (int32_t i = 0xAE; i <= 0xFF; ++i) {
            bs.push_back(static_cast<uint8_t>(i));
        }

        std::vector<int32_t> cs;
        cs.reserve(bs.size());
        for (uint8_t byte : bs) {
            cs.push_back(static_cast<int32_t>(byte));
        }

        int32_t n = 0;
        for (int32_t b = 0; b < 256; ++b) {
            uint8_t byte = static_cast<uint8_t>(b);
            if (std::find(bs.begin(), bs.end(), byte) == bs.end()) {
                bs.push_back(byte);
                cs.push_back(256 + n);
                ++n;
            }
        }

        std::unordered_map<std::string, uint8_t> result;
        for (size_t i = 0; i < cs.size(); ++i) {
            int32_t cp = cs[i];
            std::string utf8_char;

            if (cp <= 0x7F) {
                utf8_char += static_cast<char>(cp);
            } else if (cp <= 0x7FF) {
                utf8_char += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                utf8_char += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                utf8_char += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
                utf8_char += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            }

            result[utf8_char] = bs[i];
        }

        return result;
    }();

    return map;
}

int utf8_char_length(unsigned char lead_byte) {
    if ((lead_byte & 0b10000000) == 0)
        return 1;  // 0xxxxxxx (ASCII)
    else if ((lead_byte & 0b11100000) == 0b11000000)
        return 2;  // 110xxxxx
    else if ((lead_byte & 0b11110000) == 0b11100000)
        return 3;  // 1110xxxx
    else if ((lead_byte & 0b11111000) == 0b11110000)
        return 4;  // 11110xxx
    else
        return -1;  // Invalid
}

std::vector<std::string> split_utf8_chars(const std::string& input) {
    std::vector<std::string> result;
    size_t i = 0;

    while (i < input.size()) {
        unsigned char lead = static_cast<unsigned char>(input[i]);
        int len = utf8_char_length(lead);
        if (len <= 0 || i + len > input.size()) {
            std::cerr << "Invalid UTF-8 sequence at byte index " << i << std::endl;
            break;  // Stop on error
        }
        OPENVINO_ASSERT(
            std::numeric_limits<size_t>::max() - i > len,
            "UTF-8 character length exceeds size_t limit at index ", i
        );
        result.emplace_back(input.substr(i, len));
        i += len;
    }

    return result;
}

std::vector<uint8_t> apply_unicode_to_bytes(const std::string& token) {
    auto bytes_encoder = unicode_to_bytes();

    std::vector<uint8_t> res{};
    bool return_original = false;
    auto unicode_chars = split_utf8_chars(token);

    for (const auto& uni_char : unicode_chars) {
        if (bytes_encoder.count(uni_char)) {
            res.push_back(bytes_encoder.at(uni_char));
        } else {
            return_original = true;
            break;
        }
    }

    if (return_original) {
        std::vector<uint8_t> bytes(token.begin(), token.end());
        return bytes;
    }
    return res;
}

std::vector<std::vector<uint8_t>> parse_bbpe_vocab(const std::vector<std::string>& vocab, bool byte_encode = true) {
    std::vector<std::vector<uint8_t>> res;
    for (const auto& token : vocab) {
        // gpt2 uses GPT-2 unicode->byte decoding; gemma4's raw-UTF8 vocab is taken verbatim.
        res.push_back(byte_encode ? apply_unicode_to_bytes(token)
                                  : std::vector<uint8_t>(token.begin(), token.end()));
    }
    return res;
}

ov::OutputVector parse_bbpe_config(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                   ov::OutputVector inputs,
                                   const FactoryCreateType& create_func,
                                   bool byte_encode = true) {
    // 1. Parse vocab and add as input
    std::vector<std::string> vocab_from_config{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        vocab_from_config = *val;
    }
    auto vocab = parse_bbpe_vocab(vocab_from_config, byte_encode);
    auto vocab_const = create_string_constant(vocab);

    inputs.insert(inputs.end(), vocab_const.begin(), vocab_const.end());

    // 2. Parse merges
    std::vector<std::string> merges{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("merges"))) {
        merges = *val;
    }

    std::vector<std::vector<uint8_t>> left_merges;
    std::vector<std::vector<uint8_t>> right_merges;

    auto encode_piece = [&](const std::string& piece) {
        return byte_encode ? apply_unicode_to_bytes(piece) : std::vector<uint8_t>(piece.begin(), piece.end());
    };
    for (const auto& merge : merges) {
        size_t space = merge.find(' ');
        std::string left = merge.substr(0, space);
        std::string right = merge.substr(space + 1);

        left_merges.push_back(encode_piece(left));
        right_merges.push_back(encode_piece(right));
    }

    auto left_merges_const = create_string_constant(left_merges);
    auto right_merges_const = create_string_constant(right_merges);

    inputs.insert(inputs.end(), left_merges_const.begin(), left_merges_const.end());
    inputs.insert(inputs.end(), right_merges_const.begin(), right_merges_const.end());

    // 3. Extract special tokens
    ov::Tensor token_types{};
    std::vector<std::string> tokens{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    std::vector<std::vector<uint8_t>> special_tokens;
    std::vector<int32_t> special_token_indices;

    for (size_t i = 0; i < vocab.size(); ++i) {
        if (is_special_token(token_types.data<int32_t>()[i])) {
            special_tokens.push_back(vocab[i]);
            special_token_indices.push_back(static_cast<int32_t>(i));
        }
    }

    auto const_special_tokens = create_string_constant(special_tokens);
    inputs.insert(inputs.end(), const_special_tokens.begin(), const_special_tokens.end());
    ov::Output<ov::Node> const_special_token_indices =
        std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_indices.size()}, special_token_indices);
    inputs.push_back(const_special_token_indices);

    // 4. Build BPETokenizer node
    std::string unk_token = "";
    if (auto it = tokenizer_config.find("unknown_token_id");
        it != tokenizer_config.end() && std::holds_alternative<ov::Tensor>(it->second)) {
        const auto& tensor = std::get<ov::Tensor>(it->second);
        uint32_t unknown_token_id = tensor.data<uint32_t>()[0];
        unk_token = vocab_from_config[unknown_token_id];
    }

    int32_t cache_capacity =
        std::max<int32_t>(static_cast<int32_t>(vocab.size() * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY);

    std::map<std::string, ov::Any> attributes = {
        {"unk_token", unk_token},
        {"fuse_unk", true},
        {"suffix_indicator", std::string("")},
        {"end_suffix", std::string("")},
        {"byte_fallback", true},
        {"cache_capacity", cache_capacity},
    };

    return create_func("BPETokenizer", inputs, attributes);
}

// Build a minimal SentencePiece ModelProto (raw protobuf, no sentencepiece proto headers) from
// GGUF vocab arrays. Fields used: pieces (1: piece str, 2: score f32, 3: type varint),
// trainer_spec (2: model_type, byte_fallback), normalizer_spec (3: name, add_dummy_prefix).
// GGUF and SP token_type values match: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED,
// 6=BYTE. SP allows only one UNKNOWN piece; extra GGUF-type-2 entries (pad tokens) become
// USER_DEFINED.
static void pb_append_varint(std::vector<uint8_t>& buf, uint64_t value) {
    while (value >= 0x80) {
        buf.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    buf.push_back(static_cast<uint8_t>(value));
}

static void pb_append_tag_len(std::vector<uint8_t>& buf, uint32_t field, const std::vector<uint8_t>& msg) {
    pb_append_varint(buf, (static_cast<uint64_t>(field) << 3) | 2);
    pb_append_varint(buf, msg.size());
    buf.insert(buf.end(), msg.begin(), msg.end());
}

static std::vector<uint8_t> build_spm_model_proto(const std::vector<std::string>& tokens,
                                                   const std::vector<float>& scores,
                                                   const std::vector<int32_t>& token_types,
                                                   bool add_dummy_prefix,
                                                   int32_t unk_id = -1,
                                                   int32_t bos_id = -1,
                                                   int32_t eos_id = -1,
                                                   int32_t pad_id = -1) {
    std::vector<uint8_t> proto;
    // SentencePiece resolves BOS/EOS/UNK/PAD by piece string (TrainerSpec.bos_piece etc.), not by
    // id, so override those strings with the actual vocab pieces at the GGUF's token ids.
    auto piece_for = [&](int32_t id) -> std::string {
        return (id >= 0 && static_cast<size_t>(id) < tokens.size()) ? tokens[id] : std::string();
    };
    const std::string unk_piece = piece_for(unk_id);
    const std::string bos_piece = piece_for(bos_id);
    const std::string eos_piece = piece_for(eos_id);
    const std::string pad_piece = piece_for(pad_id);
    bool unk_seen = false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t gguf_type = token_types[i];
        int32_t sp_type = gguf_type;
        if (gguf_type == 2) {  // UNKNOWN
            if (!unk_seen) {
                unk_seen = true;
            } else {
                sp_type = 4;  // promote extra UNKNOWN → USER_DEFINED (pad tokens)
            }
        }
        // Build SentencePiece piece sub-message
        std::vector<uint8_t> piece_msg;
        // field 1: piece (string), tag = 0x0a
        pb_append_varint(piece_msg, (1ULL << 3) | 2);
        const auto& tok = tokens[i];
        pb_append_varint(piece_msg, tok.size());
        piece_msg.insert(piece_msg.end(), tok.begin(), tok.end());
        // field 2: score (float32), tag = 0x15
        piece_msg.push_back(0x15);
        float f = scores[i];
        uint8_t fb[4];
        std::memcpy(fb, &f, 4);
        piece_msg.insert(piece_msg.end(), fb, fb + 4);
        // field 3: type (varint), tag = 0x18
        piece_msg.push_back(0x18);
        pb_append_varint(piece_msg, static_cast<uint64_t>(sp_type));
        // Append as field 1 (pieces) of ModelProto
        pb_append_tag_len(proto, 1, piece_msg);
    }

    // trainer_spec (field 2): model_type=BPE, byte_fallback=true, special-token ids/pieces from
    // the GGUF (SentencePiece's own bos_id=1/eos_id=2/unk_id=0 defaults don't match GGUF ids).
    {
        std::vector<uint8_t> ts;
        ts.push_back(0x18);  // field 3 varint (model_type)
        pb_append_varint(ts, 2);  // BPE
        auto put_id = [&ts](uint32_t field, int32_t id) {
            if (id < 0)
                return;
            pb_append_varint(ts, (static_cast<uint64_t>(field) << 3) | 0);  // varint wire type
            pb_append_varint(ts, static_cast<uint64_t>(id));
        };
        auto put_str = [&ts](uint32_t field, const std::string& s) {
            if (s.empty())
                return;
            pb_append_varint(ts, (static_cast<uint64_t>(field) << 3) | 2);  // length-delimited
            pb_append_varint(ts, s.size());
            ts.insert(ts.end(), s.begin(), s.end());
        };
        // ids (TrainerSpec 40-43) and piece strings (45-48); bos_id() etc. resolve via the piece.
        put_id(40, unk_id);
        put_id(41, bos_id);
        put_id(42, eos_id);
        put_id(43, pad_id);
        put_str(45, unk_piece);
        put_str(46, bos_piece);
        put_str(47, eos_piece);
        put_str(48, pad_piece);
        pb_append_varint(ts, (35ULL << 3) | 0);  // field 35 varint (byte_fallback)
        ts.push_back(1);          // byte_fallback = true
        pb_append_tag_len(proto, 2, ts);
    }

    // normalizer_spec (field 3): name="identity", add_dummy_prefix (leading metaspace) from GGUF
    // tokenizer.ggml.add_space_prefix (gemma3 sets it false).
    {
        std::vector<uint8_t> ns;
        // field 1: name = "identity"
        std::string name = "identity";
        pb_append_varint(ns, (1ULL << 3) | 2);
        pb_append_varint(ns, name.size());
        ns.insert(ns.end(), name.begin(), name.end());
        // field 3: add_dummy_prefix
        pb_append_varint(ns, (3ULL << 3) | 0);
        ns.push_back(add_dummy_prefix ? 1 : 0);
        pb_append_tag_len(proto, 3, ns);
    }

    return proto;
}

// Convert SentencepieceTokenizer's sparse output (indices [N,2], values [N], dense_shape [2])
// into ragged begins/ends used by the rest of the tokenizer pipeline.
// row_splits[B+1]: row_splits[i] = number of tokens for rows 0..i-1 (cumulative).
// begins[i] = row_splits[i], ends[i] = row_splits[i+1].
static ov::OutputVector sparse_to_ragged(const ov::Output<ov::Node>& sparse_indices,
                                          const ov::Output<ov::Node>& sparse_values,
                                          const ov::Output<ov::Node>& dense_shape) {
    auto ax0_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto ax1_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto ax0_0d = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
    auto one_i64 = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});
    auto zero_i64 = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});

    // batch_indices = sparse_indices[:, 0]
    auto batch_indices = std::make_shared<v8::Gather>(sparse_indices, ax0_0d, ax1_1d);
    // B = dense_shape[0] (batch size)
    auto B = std::make_shared<v8::Gather>(dense_shape, ax0_0d, ax0_0d);
    auto B_plus_1 = std::make_shared<v1::Add>(B, one_i64);
    // range [0, B+1) as row boundaries
    auto range = std::make_shared<v4::Range>(zero_i64, B_plus_1, one_i64, element::i64);
    // mask[n, i] = batch_indices[n] < range[i]  → row_splits[i] = sum of mask[:, i]
    auto bi_unsq = std::make_shared<v0::Unsqueeze>(batch_indices, ax1_1d);
    auto range_unsq = std::make_shared<v0::Unsqueeze>(range, ax0_1d);
    auto mask = std::make_shared<v1::Less>(bi_unsq, range_unsq);
    auto mask_i32 = std::make_shared<v0::Convert>(mask, element::i32);
    auto row_splits = std::make_shared<v1::ReduceSum>(mask_i32, ax0_1d, false)->output(0);

    // begins = row_splits[0:B], ends = row_splits[1:B+1]
    auto one_i64_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto zero_i64_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto int64_max = std::make_shared<v0::Constant>(element::i64, Shape{1},
                                                     std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    // B as a 1D i64 tensor (v8::Slice stop must be 1D)
    auto B_i64_1d = std::make_shared<v0::Unsqueeze>(B, ax0_1d);
    auto begins = std::make_shared<v8::Slice>(row_splits, zero_i64_1d, B_i64_1d, one_i64_1d, ax0_1d)->output(0);
    auto ends_node = std::make_shared<v8::Slice>(row_splits, one_i64_1d, int64_max, one_i64_1d, ax0_1d)->output(0);

    auto values_i32 = std::make_shared<v0::Convert>(sparse_values, element::i32)->output(0);
    return {begins, ends_node, values_i32};
}

// Scalar bool/id lookups over the GGUF tokenizer metadata. Values are typically a scalar Tensor;
// a plain int is also accepted.
static bool read_tokenizer_flag(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                const char* key,
                                bool dflt) {
    if (auto it = tokenizer_config.find(key); it != tokenizer_config.end()) {
        if (std::holds_alternative<ov::Tensor>(it->second)) {
            const auto& t = std::get<ov::Tensor>(it->second);
            if (t.get_size() > 0)
                return t.data<bool>()[0];
        } else if (std::holds_alternative<int>(it->second)) {
            return std::get<int>(it->second) != 0;
        }
    }
    return dflt;
}

static int32_t read_tokenizer_id(const std::map<std::string, GGUFMetaData>& tokenizer_config, const char* key) {
    if (auto it = tokenizer_config.find(key); it != tokenizer_config.end()) {
        if (std::holds_alternative<ov::Tensor>(it->second)) {
            const auto& t = std::get<ov::Tensor>(it->second);
            if (t.get_size() > 0)
                return static_cast<int32_t>(t.data<uint32_t>()[0]);
        } else if (std::holds_alternative<int>(it->second)) {
            return static_cast<int32_t>(std::get<int>(it->second));
        }
    }
    return -1;
}

// Build SentencePiece (model="llama") tokenizer outputs from a serialized ModelProto via the
// SentencepieceTokenizer op.
static ov::OutputVector parse_spm_config(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                          ov::OutputVector inputs,
                                          const FactoryCreateType& create_func) {
    std::vector<std::string> vocab;
    ov::Tensor token_types_tensor;

    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens")))
        vocab = *val;
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type")))
        token_types_tensor = *val;

    ov::Tensor scores_tensor;
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("scores")))
        scores_tensor = *val;

    OPENVINO_ASSERT(!vocab.empty(), "[gguf tokenizer] SentencePiece: 'tokens' array is missing or empty");
    OPENVINO_ASSERT(scores_tensor.get_size() == vocab.size(),
                    "[gguf tokenizer] SentencePiece: 'scores' tensor size (", scores_tensor.get_size(),
                    ") != vocab size (", vocab.size(), ")");
    OPENVINO_ASSERT(token_types_tensor.get_size() == vocab.size(),
                    "[gguf tokenizer] SentencePiece: 'token_type' array size != vocab size");

    std::vector<float> scores(scores_tensor.data<float>(),
                              scores_tensor.data<float>() + scores_tensor.get_size());
    std::vector<int32_t> token_types(token_types_tensor.data<int32_t>(),
                                     token_types_tensor.data<int32_t>() + token_types_tensor.get_size());

    // SentencePiece requires exactly one UNKNOWN token (type=2). Some models (gemma2, gemma3)
    // have no type-2 token in their GGUF vocab but do supply unknown_token_id. Promote it.
    if (std::none_of(token_types.begin(), token_types.end(), [](int32_t t) { return t == 2; })) {
        if (auto it = tokenizer_config.find("unknown_token_id"); it != tokenizer_config.end()) {
            if (auto t = std::get_if<ov::Tensor>(&it->second)) {
                const uint32_t unk_id = t->data<uint32_t>()[0];
                if (unk_id < token_types.size())
                    token_types[unk_id] = 2;
            }
        }
    }

    // add_space_prefix (GGUF bool, default true): leading metaspace prefix; gemma3 sets it false.
    bool add_space_prefix = true;
    if (auto it = tokenizer_config.find("add_space_prefix");
        it != tokenizer_config.end() && std::holds_alternative<ov::Tensor>(it->second)) {
        const auto& t = std::get<ov::Tensor>(it->second);
        if (t.get_size() > 0)
            add_space_prefix = t.data<bool>()[0];
    }

    // Special-token ids from the GGUF, baked into the SP proto's trainer_spec.
    const int32_t unk_id = read_tokenizer_id(tokenizer_config, "unknown_token_id");
    const int32_t bos_id = read_tokenizer_id(tokenizer_config, "bos_token_id");
    const int32_t eos_id = read_tokenizer_id(tokenizer_config, "eos_token_id");
    const int32_t pad_id = read_tokenizer_id(tokenizer_config, "padding_token_id");

    // Build the serialized ModelProto and wrap as a u8 Constant (first input to SentencepieceTokenizer)
    auto proto_bytes =
        build_spm_model_proto(vocab, scores, token_types, add_space_prefix, unk_id, bos_id, eos_id, pad_id);
    auto sp_model_const = std::make_shared<v0::Constant>(element::u8, Shape{proto_bytes.size()}, proto_bytes.data());

    // inputs = SpecialTokensSplit outputs: [ragged_begins(0), ragged_ends(1), begins(2), ends(3), chars(4), ...]
    // SentencepieceTokenizer with 4 inputs: (sp_model, begins, ends, chars) — the inner flat ragged string.
    OPENVINO_ASSERT(inputs.size() >= 5, "[gguf tokenizer] SentencePiece: expected >=5 outputs from SpecialTokensSplit");
    ov::OutputVector sp_inputs = {sp_model_const->output(0), inputs[2], inputs[3], inputs[4]};
    // BOS/EOS are not baked into SentencepieceTokenizer: its add_bos/add_eos are compile-time
    // attributes that add_special_tokens=false can't switch off. Emit them as a CombineSegments
    // segment instead, like the BPE paths (see the CombineSegments block below).
    auto sp_tok = create_func("SentencepieceTokenizer", sp_inputs,
                               {{"nbest_size", int32_t{0}},
                                {"alpha", 0.0f},
                                {"add_bos", false},
                                {"add_eos", false},
                                {"reverse", false}});
    // sp_tok = [sparse_indices [N,2] i64, sparse_values [N] i32, dense_shape [2] i64]
    OPENVINO_ASSERT(sp_tok.size() == 3, "[gguf tokenizer] SentencepieceTokenizer must have 3 outputs");
    // SP tokenizes each inner segment independently; map its sparse output to an inner ragged
    // form, then FuzeRagged collapses back to batch-level using the outer ragged begins/ends.
    auto inner = sparse_to_ragged(sp_tok[0], sp_tok[1], sp_tok[2]);
    // inner = [begins(N_inner), ends(N_inner), values]
    auto fused = create_func("FuzeRagged", {inputs[0], inputs[1], inner[0], inner[1]}, {});
    // fused = [batch_begins(batch), batch_ends(batch)]
    return {fused[0], fused[1], inner[2]};
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
build_tokenizer_models(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                       std::map<std::string, GGUFMetaData> tokenizer_config) {
    auto tokenizer_input = std::make_shared<v0::Parameter>(element::string, PartialShape{Dimension::dynamic()});

    FactoryCreateType create_func =
        reinterpret_cast<FactoryCreateType>(get_symbol(shared_object_ov_tokenizers, "create_tokenizer_node"));

    std::string model{};
    if (auto val = std::get_if<std::string>(&tokenizer_config.at("model"))) {
        model = *val;
    }

    OutputVector outputs = create_func("StringTensorUnpack", {tokenizer_input}, {});

    // gemma4 is an SPM-style BPE: escape whitespace to U+2581 (metaspace) on the raw decomposed
    // string (indices 0,1,2 = begins/ends/chars) before the ragged dimension is added. It does
    // NOT prepend a leading metaspace (llama.cpp gemma4: add_space_prefix=false), so only the
    // space->metaspace replacement is applied.
    if (model == "gemma4") {
        const std::string metaspace = "\xe2\x96\x81";  // U+2581 ▁
        auto make_str_scalar = [](const std::string& s) {
            ov::Tensor t(ov::element::u8, {s.size()});
            std::memcpy(t.data<uint8_t>(), s.data(), s.size());
            return std::make_shared<v0::Constant>(t)->output(0);
        };
        ov::OutputVector in(outputs.begin(), outputs.begin() + 3);
        in.push_back(make_str_scalar(" "));
        in.push_back(make_str_scalar(metaspace));
        auto normed = create_func("RegexNormalization", in, {{"global_replace", true}});
        for (size_t i = 0; i < normed.size() && i < outputs.size(); ++i) {
            outputs[i] = normed[i];
        }
    }

    outputs = add_ragged_dimension(outputs);

    // Special token filtering
    std::vector<std::string> tokens, special_tokens;
    ov::Tensor token_types;
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    const auto token_types_data = token_types.data<int32_t>();

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special_token(token_types_data[i])) {
            special_tokens.push_back(tokens[i]);
        }
    }

    std::string special_tokens_re = join_special_tokens(special_tokens);

    ov::Tensor ov_special_tokens(ov::element::u8, {special_tokens_re.size()});
    std::memcpy(ov_special_tokens.data<uint8_t>(), special_tokens_re.data(), special_tokens_re.size());
    auto const_special_tokens = std::make_shared<v0::Constant>(ov_special_tokens);

    ov::OutputVector inputs_to_split = outputs;
    inputs_to_split.push_back(const_special_tokens->output(0));
    outputs = create_func("SpecialTokensSplit", inputs_to_split, {});

    // plamo2 uses the same SentencePiece BPE format as llama
    const std::string effective_model = (model == "plamo2") ? "llama" : model;

    OPENVINO_ASSERT(effective_model == "gpt2" || effective_model == "llama" || effective_model == "gemma4",
                    "[gguf tokenizer] Unsupported tokenizer model '", model,
                    "'. Supported: 'gpt2' (BPE), 'llama'/'plamo2' (SentencePiece BPE), 'gemma4' (SPM-style BPE).");

    if (effective_model == "llama") {
        // SentencePiece: SP handles word splitting internally; skip BPE-style RegexSplit.
        // outputs[0..2] = ragged string (begins, ends, chars) from SpecialTokensSplit.
        outputs = parse_spm_config(tokenizer_config, outputs, create_func);
        // outputs = [begins, ends, values] (ragged token ids, i32)
    } else {
        // BPE: gpt2 is byte-level BPE with regex pre-tokenization; gemma4 is SPM-style BPE over
        // the raw-UTF8 vocab, split only on newlines (mirrors llama.cpp's GEMMA4 pre-type).
        if (model == "gemma4") {
            const std::string newline_split = "[^\\n]+|[\\n]+";
            ov::Tensor ov_split_re(ov::element::u8, {newline_split.size()});
            std::memcpy(ov_split_re.data<uint8_t>(), newline_split.data(), newline_split.size());
            auto const_ov_split_re = std::make_shared<v0::Constant>(ov_split_re);
            outputs.push_back(const_ov_split_re->output(0));
            outputs = create_func("RegexSplit", outputs,
                                   {{"behaviour", std::string("isolate")}, {"invert", false}, {"max_splits", -1}});

            ov::OutputVector bbpe_inputs(outputs.begin(), outputs.begin() + 5);
            outputs = parse_bbpe_config(tokenizer_config, bbpe_inputs, create_func, /*byte_encode=*/false);
        } else {
            // BPE (gpt2): apply regex pre-tokenization splits before the BPE encoder.
            std::string pre{};
            if (auto val = std::get_if<std::string>(&tokenizer_config.at("pre"))) {
                pre = *val;
            }
            for (const auto& split_re : get_split_regex(pre)) {
                ov::Tensor ov_split_re(ov::element::u8, {split_re.size()});
                std::memcpy(ov_split_re.data<uint8_t>(), split_re.data(), split_re.size());
                auto const_ov_split_re = std::make_shared<v0::Constant>(ov_split_re);
                outputs.push_back(const_ov_split_re->output(0));
                outputs = create_func("RegexSplit", outputs,
                                      {{"behaviour", std::string("isolate")}, {"invert", false}, {"max_splits", -1}});
            }
            ov::OutputVector bbpe_inputs(outputs.begin(), outputs.begin() + 5);
            outputs = parse_bbpe_config(tokenizer_config, bbpe_inputs, create_func);
        }
    }

    ov::Output<ov::Node> max_length = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, MAX_LENGTH);
    ov::Output<ov::Node> ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    max_length = std::make_shared<v1::Minimum>(ends_minus_begins, max_length);
    outputs[0] = std::make_shared<v1::Subtract>(outputs[1], max_length)->output(0);

    // BOS/EOS as an explicit CombineSegments segment, for every tokenizer flavour: BPE has no
    // add_bos mechanism at all, and SentencepieceTokenizer's add_bos/add_eos are compile-time
    // attributes, so neither backend can honor encode(add_special_tokens=false) on its own.
    //
    // Must come after the truncation above: MakeAddSpecialTokensSatateful identifies the
    // main-sequence segment by its begins being a Subtract, and toggles every other segment's
    // `ends` at runtime; emitting this before truncation would break that detection.
    {
        // Defaults follow llama.cpp (llama-vocab.cpp): SPM defaults add_bos true; for BPE the base
        // default is false, but a few pre-tokenizer families flip it to true and gemma4 is forced
        // true. An explicit tokenizer.ggml.add_bos_token in the GGUF always wins over the default.
        static const std::set<std::string> bpe_pre_add_bos = {
            // LLAMA_VOCAB_PRE_TYPE_LLAMA3 group
            "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0",
            "lfm2", "jina-v5-nano",
            // standalone
            "tekken", "chameleon",
        };
        std::string pre{};
        if (auto it = tokenizer_config.find("pre"); it != tokenizer_config.end()) {
            if (auto val = std::get_if<std::string>(&it->second)) {
                pre = *val;
            }
        }
        const bool default_add_bos = (effective_model == "llama") || (model == "gemma4") ||
                                     bpe_pre_add_bos.count(pre) > 0;

        const bool add_bos = read_tokenizer_flag(tokenizer_config, "add_bos_token", default_add_bos);
        const bool add_eos = read_tokenizer_flag(tokenizer_config, "add_eos_token", false);
        const int32_t bos_id = read_tokenizer_id(tokenizer_config, "bos_token_id");
        const int32_t eos_id = read_tokenizer_id(tokenizer_config, "eos_token_id");

        ov::OutputVector cs_inputs;
        std::vector<int32_t> segment_ids;
        // One added-token segment is [begins=0, ends=1, ids=[tok]]; `ends` is what the stateful
        // pass rewrites to 0 when the caller asks for add_special_tokens=false.
        auto add_token_segment = [&](int32_t token_id) {
            cs_inputs.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0)->output(0));
            cs_inputs.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 1)->output(0));
            cs_inputs.push_back(
                std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{token_id})
                    ->output(0));
            segment_ids.push_back(0);
        };

        if (add_bos && bos_id >= 0) {
            add_token_segment(bos_id);
        }
        cs_inputs.push_back(outputs[0]);
        cs_inputs.push_back(outputs[1]);
        cs_inputs.push_back(outputs[2]);
        segment_ids.push_back(0);
        if (add_eos && eos_id >= 0) {
            add_token_segment(eos_id);
        }

        if (segment_ids.size() > 1) {
            cs_inputs.push_back(
                std::make_shared<v0::Constant>(element::i32, ov::Shape{segment_ids.size()}, segment_ids)->output(0));
            auto combined = create_func("CombineSegments", cs_inputs, {});
            // CombineSegments returns [begins, ends, values, ...]; keep the ragged token ids.
            OPENVINO_ASSERT(combined.size() >= 3, "[gguf tokenizer] CombineSegments must have >=3 outputs");
            outputs[0] = combined[0];
            outputs[1] = combined[1];
            outputs[2] = combined[2];
        }
    }

    // Left padding
    ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    auto reduce_axis = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    ov::Output<ov::Node> max_length_batch = std::make_shared<v1::ReduceMax>(ends_minus_begins, reduce_axis, false);

    ov::OutputVector inputs_for_ragged_to_dense = outputs;
    inputs_for_ragged_to_dense.push_back(max_length_batch);
    ov::Output<ov::Node> const_zero_for_rg = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0);
    inputs_for_ragged_to_dense.push_back(const_zero_for_rg);

    outputs =
        create_func("RaggedToDense", inputs_for_ragged_to_dense, {{"pad_right", false}, {"pad_max_length", false}});

    // Convert output types
    outputs[0] = std::make_shared<v0::Convert>(outputs[0], element::i64)->output(0);
    outputs[1] = std::make_shared<v0::Convert>(outputs[1], element::i64)->output(0);
    outputs[0].get_tensor().add_names({"input_ids"});
    outputs[1].get_tensor().add_names({"attention_mask"});

    auto tokenizer = std::make_shared<Model>(outputs, ParameterVector{tokenizer_input}, "tokenizer");

    // DETOKENIZER model
    auto detokenizer_input =
        std::make_shared<v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});

    std::shared_ptr<Model> detokenizer;
    if (effective_model == "llama") {
        // SentencepieceDetokenizer: takes (sp_model_u8, token_ids_i32) → ragged string
        std::vector<int32_t> spm_types(token_types.data<int32_t>(),
                                       token_types.data<int32_t>() + token_types.get_size());
        // Same unk-promotion as the tokenizer: ensure exactly one type-2 entry.
        if (std::none_of(spm_types.begin(), spm_types.end(), [](int32_t t) { return t == 2; })) {
            if (auto it = tokenizer_config.find("unknown_token_id"); it != tokenizer_config.end()) {
                if (auto t = std::get_if<ov::Tensor>(&it->second)) {
                    const uint32_t unk_id = t->data<uint32_t>()[0];
                    if (unk_id < spm_types.size())
                        spm_types[unk_id] = 2;
                }
            }
        }
        std::vector<float> spm_scores;
        if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("scores"))) {
            const ov::Tensor& st = *val;
            spm_scores.assign(st.data<float>(), st.data<float>() + st.get_size());
        }
        // Mirror the tokenizer's add_space_prefix so detokenization strips the leading
        // metaspace symmetrically (gemma3: false).
        bool detok_add_space_prefix = true;
        if (auto it = tokenizer_config.find("add_space_prefix");
            it != tokenizer_config.end() && std::holds_alternative<ov::Tensor>(it->second)) {
            const auto& t = std::get<ov::Tensor>(it->second);
            if (t.get_size() > 0)
                detok_add_space_prefix = t.data<bool>()[0];
        }
        auto proto_bytes = build_spm_model_proto(tokens, spm_scores, spm_types, detok_add_space_prefix);
        auto sp_model_const =
            std::make_shared<v0::Constant>(element::u8, Shape{proto_bytes.size()}, proto_bytes.data());
        auto ids_i32 = std::make_shared<v0::Convert>(detokenizer_input, element::i32)->output(0);
        auto sp_detok = create_func("SentencepieceDetokenizer",
                                    {sp_model_const->output(0), ids_i32},
                                    {});
        // sp_detok outputs = ragged string (begins, ends, chars) → pack → string tensor
        auto packed = create_func("StringTensorPack", sp_detok, {});
        packed[0].get_tensor().add_names({"string_output"});
        detokenizer = std::make_shared<Model>(packed, ParameterVector{detokenizer_input}, "detokenizer");
    } else {
        // BPE detokenizer: VocabDecoder + FuzeRagged + UTF8Validate. gpt2 uses GPT-2 byte
        // decoding of the vocab; gemma4 uses the raw-UTF8 vocab (and undoes the metaspace
        // afterwards, below).
        const bool byte_encode = (model != "gemma4");
        auto vocab = parse_bbpe_vocab(tokens, byte_encode);
        ov::OutputVector const_vocab = create_string_constant(vocab);
        OutputVector detokenizer_outputs = {detokenizer_input};
        detokenizer_outputs.insert(detokenizer_outputs.end(), const_vocab.begin(), const_vocab.end());

        std::vector<int32_t> special_token_ids;
        for (size_t i = 0; i < token_types.get_size(); ++i) {
            if (is_special_token(token_types.data<int32_t>()[i]))
                special_token_ids.push_back(static_cast<int32_t>(i));
        }

        auto special_ids_const =
            std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_ids.size()}, special_token_ids);
        auto const_zero = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
        auto const_one = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 1);
        int32_t int32_max_value = std::numeric_limits<int32_t>::max();
        auto const_int32_max = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, int32_max_value);
        auto sliced_skips =
            std::make_shared<v8::Slice>(special_ids_const, const_zero, const_int32_max, const_one)->outputs();
        detokenizer_outputs.insert(detokenizer_outputs.end(), sliced_skips.begin(), sliced_skips.end());

        detokenizer_outputs = create_func("VocabDecoder", detokenizer_outputs, {});
        ov::OutputVector inputs_for_fused_ragged(detokenizer_outputs.begin(), detokenizer_outputs.end() - 1);
        auto outputs_fused_ragged = create_func("FuzeRagged", inputs_for_fused_ragged, {});
        outputs_fused_ragged.insert(outputs_fused_ragged.end(), detokenizer_outputs.end() - 1, detokenizer_outputs.end());
        ov::OutputVector inputs_for_utf8_validate(outputs_fused_ragged.begin(), outputs_fused_ragged.end());
        auto outputs_utf8_validate =
            create_func("UTF8Validate", inputs_for_utf8_validate, {{"replace_mode", true}});
        if (model == "gemma4") {
            // Undo the metaspace: U+2581 -> space, on the decoded ragged string.
            const std::string metaspace = "\xe2\x96\x81";
            auto make_str_scalar = [](const std::string& s) {
                ov::Tensor t(ov::element::u8, {s.size()});
                std::memcpy(t.data<uint8_t>(), s.data(), s.size());
                return std::make_shared<v0::Constant>(t)->output(0);
            };
            ov::OutputVector norm_inputs(outputs_utf8_validate.begin(),
                                         outputs_utf8_validate.begin() + 3);
            norm_inputs.push_back(make_str_scalar(metaspace));
            norm_inputs.push_back(make_str_scalar(" "));
            outputs_utf8_validate = create_func("RegexNormalization", norm_inputs, {{"global_replace", true}});
        }
        auto packed_output = create_func("StringTensorPack", outputs_utf8_validate, {});
        packed_output[0].get_tensor().add_names({"string_output"});
        detokenizer = std::make_shared<Model>(packed_output, ParameterVector{detokenizer_input}, "detokenizer");
    }

    return {tokenizer, detokenizer, tokenizer_config};
}

// Convert the frontend's rt_info tokenizer metadata into the GGUFMetaData config map.
std::map<std::string, GGUFMetaData> tokenizer_config_from_rt_info(const ov::AnyMap& rt_cfg) {
    std::map<std::string, GGUFMetaData> cfg;
    for (const auto& [key, value] : rt_cfg) {
        if (value.is<std::string>()) {
            cfg[key] = value.as<std::string>();
        } else if (value.is<std::vector<std::string>>()) {
            cfg[key] = value.as<std::vector<std::string>>();
        } else if (value.is<ov::Tensor>()) {
            cfg[key] = value.as<ov::Tensor>();
        } else if (value.is<int>()) {
            cfg[key] = value.as<int>();
        } else if (value.is<float>()) {
            cfg[key] = value.as<float>();
        }
        // other types are not part of the tokenizer config and are ignored
    }
    return cfg;
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_config(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                             const std::filesystem::path& gguf_model_path) {
    auto gguf_metadata = get_gguf_metadata(gguf_model_path.string());
    auto tokenizer_config = tokenizer_config_from_meta(gguf_metadata);
    return build_tokenizer_models(shared_object_ov_tokenizers, std::move(tokenizer_config));
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_parameters(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                                 const ov::AnyMap& tokenizer_metadata) {
    OPENVINO_ASSERT(!tokenizer_metadata.empty(),
                    "[gguf tokenizer] empty GGUF tokenizer metadata: there is nothing to build a tokenizer from.");
    auto tokenizer_config = tokenizer_config_from_rt_info(tokenizer_metadata);
    return build_tokenizer_models(shared_object_ov_tokenizers, std::move(tokenizer_config));
}

ov::AnyMap gguf_tokenizer_metadata_from_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "[gguf tokenizer] null model: cannot read GGUF tokenizer metadata from it.");
    const auto& rt = model->get_rt_info();
    auto it = rt.find(ov::frontend::gguf::gguf_tokenizer_metadata_key());
    OPENVINO_ASSERT(it != rt.end(),
                    "[gguf tokenizer] the model has no '",
                    ov::frontend::gguf::gguf_tokenizer_metadata_key(),
                    "' runtime info, so no tokenizer can be built from it. Only a model converted from a .gguf "
                    "by the OpenVINO GGUF frontend carries it; a plain IR, a model whose rt_info was stripped, "
                    "or one built by the legacy GGUF reader does not. Build the Tokenizer from the .gguf path "
                    "instead.");
    auto attr = it->second.as<std::shared_ptr<ov::frontend::gguf::GGUFTokenizerMetadata>>();
    OPENVINO_ASSERT(attr, "[gguf tokenizer] unexpected type for the GGUF tokenizer metadata runtime info.");
    return attr->config;
}

std::string patch_gguf_chat_template(const std::string& chat_template) {
    std::string patched_chat_template = chat_template;
    // Define the exact pattern to find in original chat_template
    // Using C++ raw string literals (R"(...)") to correctly represent the literal content,
    const std::string qwen2_5_substring_to_find = R"({{\"name\": <function-name>, \"arguments\": <args-json-object>}})";
    // Define the exact replacement substring for str2
    const std::string qwen2_5_replacement_substring =
        R"({\"name\": <function-name>, \"arguments\": <args-json-object>})";
    // Find the position of the substring to be replaced
    size_t pos_qwen2_5 = patched_chat_template.find(qwen2_5_substring_to_find);
    if (pos_qwen2_5 != std::string::npos) {
        // Substring found, perform the replacement
        patched_chat_template.replace(pos_qwen2_5, qwen2_5_substring_to_find.length(), qwen2_5_replacement_substring);
    }

    const std::string qwen3_substring_to_find_0 = R"({%- for index in range(ns.last_query_index, -1, -1) %})";
    const std::string qwen3_substring_to_find_1 = R"({%- set message = messages[index] %})";
    const std::string qwen3_substring_to_find_2 = R"({%- if ns.multi_step_tool and message.role == "user" and not('<tool_response>' in message.content and '</tool_response>' in message.content) %})";

    const std::string qwen3_replacement_substring_0 = R"({%- for message in messages[::-1] %})";
    const std::string qwen3_replacement_substring_1 = R"({%- set index = (messages|length - 1) - loop.index0 %})";
    const std::string qwen3_replacement_substring_2 = R"({%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %})";

    const std::string qwen3_substring_to_find = qwen3_substring_to_find_0 + "\n" + "    " + qwen3_substring_to_find_1 + "\n" + "    "  + qwen3_substring_to_find_2;
    const std::string qwen3_replacement_substring = qwen3_replacement_substring_0 + "\n" + "    " + qwen3_replacement_substring_1 + "\n" + "    "  + qwen3_replacement_substring_2;
    size_t pos_qwen3 = patched_chat_template.find(qwen3_substring_to_find);
    
    if (pos_qwen3 != std::string::npos) {
        // Substring found, perform the replacement
        patched_chat_template.replace(pos_qwen3, qwen3_substring_to_find.length(), qwen3_replacement_substring);
    }

    return patched_chat_template;
}

}  // namespace genai
}  // namespace ov
