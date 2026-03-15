#include "misaki/g2p.hpp"

#include <iostream>
#include <string>

int main() {
  auto engine = misaki::make_engine("en", "en-us");

  const std::string text = "Misaki consumer sample";
  const auto result = engine->phonemize_with_tokens(text);

  std::cout << "Input:    " << text << "\n";
  std::cout << "Phonemes: " << result.phonemes << "\n";
  std::cout << "Tokens:   " << result.tokens.size() << "\n";

  return 0;
}
