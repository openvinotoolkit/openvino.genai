#include "qwen.h"
#include "base64.h"
#include "unordered_dense.h"
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <random>
#include <thread>
#include <sys/stat.h>
#include <iostream>
#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

namespace qwen {
// ===== streamer =====
auto TextStreamer::put(const std::vector<int> &output_ids) -> void {
  if (is_prompt_) {
    is_prompt_ = false;
    return;
  }

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};

  token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
  std::string text = tokenizer_->decode(token_cache_);
  if (text.empty()) {
    return;
  }

  std::string printable_text;
  if (text.back() == '\n') {
    // flush the cache after newline
    printable_text = text.substr(print_len_);
    token_cache_.clear();
    print_len_ = 0;
  } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
    // last symbol is a punctuation, hold on
  } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
    // ends with an incomplete token, hold on
  } else {
    printable_text = text.substr(print_len_);
    print_len_ = text.size();
  }

  os_ << printable_text << std::flush;
}

auto TextStreamer::end() -> void {
  std::string text = tokenizer_->decode(token_cache_);
  os_ << text.substr(print_len_) << std::endl;
  is_prompt_ = true;
  token_cache_.clear();
  print_len_ = 0;
}

// ===== Qwen Tokenizer =====

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

QwenTokenizer::QwenTokenizer(const std::string & tiktoken_path, const QwenConfig &config) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
  eos_token_id = config.eos_token_id;
  im_start_id = config.im_start_id;
  im_end_id = config.im_end_id;
}

auto QwenTokenizer::build_prompt(const std::vector<std::string> &history) const -> std::string {
  QWEN_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

  std::ostringstream oss_prompt;
  oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
  for (size_t i = 0; i < history.size() - 1; i += 2) {
    oss_prompt << "\n<|im_start|>user\n" << history[i] << "<|im_end|>\n<|im_start|>" << history[i + 1] << "<|im_end|>";
  }
  oss_prompt << "\n<|im_start|>user\n" << history.back() <<  "<|im_end|>\n<|im_start|>assistant\n";

  return oss_prompt.str();
}

auto QwenTokenizer::encode(const std::string &text, int max_length) const -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

auto QwenTokenizer::encode_history(
  const std::vector<std::string> &history, int max_length
) const -> std::vector<int> {
  std::string prompt = build_prompt(history);
  std::vector<int> input_ids = encode(prompt, max_length);
  return input_ids;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

} // namespace qwen
