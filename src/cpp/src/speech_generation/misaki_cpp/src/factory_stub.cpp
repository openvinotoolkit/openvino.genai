#include "misaki/g2p.hpp"
#include <stdexcept>

namespace misaki {

std::unique_ptr<G2P> make_espeak_engine(const std::string& variant);

std::unique_ptr<G2P> make_engine(const std::string& lang, const std::string& variant) {
  if (lang == "en") {
    return make_english_engine(variant);
  }
  if (lang == "espeak") {
    return make_espeak_engine(variant);
  }
  throw std::runtime_error(
      "Unsupported language: " + lang +
      " (supported: 'en' with variants 'en-us'/'en-gb', 'espeak' with variants 'es'/'fr-fr'/'hi'/'it'/'pt-br')");
}

} // namespace misaki
