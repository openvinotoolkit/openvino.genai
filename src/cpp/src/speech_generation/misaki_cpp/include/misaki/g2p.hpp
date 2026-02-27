#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace misaki {

struct Request {
  std::string lang;
  std::string variant;
  std::string text;
};

struct MToken {
  struct Underscore {
    std::optional<bool> is_head;
    std::optional<std::string> alias;
    std::optional<double> stress;
    std::optional<std::string> currency;
    std::string num_flags;
    std::optional<bool> prespace;
    std::optional<int> rating;
  };

  std::string text;
  std::string tag;
  std::string whitespace;
  std::optional<std::string> phonemes;
  std::optional<double> start_ts;
  std::optional<double> end_ts;
  std::optional<Underscore> _;
};

struct PhonemizeResult {
  std::string phonemes;
  std::vector<MToken> tokens;
};

class G2P {
public:
  using FallbackHook = std::function<std::optional<std::string>(const MToken&)>;

  virtual ~G2P() = default;
  virtual PhonemizeResult phonemize_with_tokens(const std::string& text) const = 0;

  void set_fallback_hook(FallbackHook hook) {
    fallback_hook_ = std::move(hook);
  }

  void set_unknown_token(std::string unknown_token) {
    unknown_token_ = std::move(unknown_token);
  }

  std::string phonemize(const std::string& text) const {
    return phonemize_with_tokens(text).phonemes;
  }

protected:
  std::optional<std::string> run_fallback_hook(const MToken& token) const {
    if (!fallback_hook_) {
      return std::nullopt;
    }
    return fallback_hook_(token);
  }

  const std::string& unknown_token() const {
    return unknown_token_;
  }

private:
  FallbackHook fallback_hook_;
  std::string unknown_token_ = "❓";
};

std::unique_ptr<G2P> make_english_engine(const std::string& variant);
std::unique_ptr<G2P> make_engine(const std::string& lang, const std::string& variant);

void set_english_lexicon_data_root(const std::string& path);
void clear_english_lexicon_data_root();

} // namespace misaki
