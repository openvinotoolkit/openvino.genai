#include "misaki/g2p.hpp"
#include "misaki/fallbacks.hpp"

#include <stdexcept>
#include <utility>

namespace misaki {
namespace {

class EspeakLanguageG2P final : public G2P {
public:
  explicit EspeakLanguageG2P(std::string language, std::string version = {})
      : language_(std::move(language)), backend_(language_, std::move(version)) {}

  PhonemizeResult phonemize_with_tokens(const std::string &text) const override {
    PhonemizeResult out;

    const auto phonemes = backend_.phonemize(text);
    if (phonemes.has_value()) {
      out.phonemes = *phonemes;
    } else {
      out.phonemes.clear();
    }
    return out;
  }

private:
  std::string language_;
  EspeakG2P backend_;
};

bool is_supported_espeak_language(const std::string &variant) {
  return variant == "es" || variant == "fr-fr" || variant == "hi" || variant == "it" || variant == "pt-br";
}

} // namespace

std::unique_ptr<G2P> make_espeak_engine(const std::string &variant) {
  if (!is_supported_espeak_language(variant)) {
    throw std::runtime_error(
        "Unsupported Espeak G2P variant: " + variant +
        " (supported: es, fr-fr, hi, it, pt-br)");
  }
  return std::make_unique<EspeakLanguageG2P>(variant);
}

} // namespace misaki
