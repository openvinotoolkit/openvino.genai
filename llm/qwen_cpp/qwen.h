#pragma once

#include "tiktoken.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen {

class QwenTokenizer;

// ===== common =====

static constexpr size_t MB = 1024 * 1024;

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

class LogMessageFatal {
  public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    auto stream() -> std::ostringstream & { return oss_; }

  private:
    std::ostringstream oss_;
};

#define QWEN_THROW ::qwen::LogMessageFatal(__FILE__, __LINE__).stream()
#define QWEN_CHECK(cond) \
    if (!(cond)) \
    QWEN_THROW << "check failed (" #cond ") "
class BaseStreamer{
  public:
    virtual ~BaseStreamer() = default;
    virtual auto put(const std::vector<int> &output_ids) -> void = 0;
    virtual auto end() -> void = 0;
};

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
class TextStreamer : public BaseStreamer {
  public:
    TextStreamer(std::ostream &os, QwenTokenizer *tokenizer)
        : os_(os), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    auto put(const std::vector<int> &output_ids) -> void override;
    auto end() -> void override;

  private:
    std::ostream &os_;
    QwenTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

// ===== Qwen-7B =====

struct QwenConfig {
  // common attributes
  // ggml_type dtype;
  int vocab_size = 151936;
  int hidden_size = 4096;
  int num_attention_heads = 32;
  int num_kv_heads = 32;
  int num_hidden_layers = 32;
  int intermediate_size = 22016;
  // for sequence generation
  int max_length = 8192;
  // for tokenizer
  int eos_token_id = 151643;
  int pad_token_id = 151643;
  int im_start_id = 151644;
  int im_end_id = 151645;
};

class QwenTokenizer {
  public:

    QwenTokenizer(const std::string & tiktoken_path, const QwenConfig &config);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int>;

    auto build_prompt(const std::vector<std::string> &history) const -> std::string;

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;
};
} // namespace qwen
