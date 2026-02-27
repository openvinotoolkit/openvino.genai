#include "misaki/g2p.hpp"
#include <stdexcept>

namespace misaki {

std::unique_ptr<G2P> make_engine(const std::string& lang, const std::string& variant) {
  if (lang == "en") {
    return make_english_engine(variant);
  }
  throw std::runtime_error(
      "Unsupported language: " + lang + " (only 'en' with variants 'en-us' or 'en-gb' is supported)");
}

} // namespace misaki
