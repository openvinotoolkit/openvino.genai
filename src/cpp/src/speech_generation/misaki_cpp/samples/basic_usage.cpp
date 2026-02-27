#include "misaki/g2p.hpp"

#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
void configure_utf8_console() {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
}
} // namespace

int main() {
  configure_utf8_console();

  auto engine = misaki::make_engine("en", "en-us");

  //const std::string text = "[Misaki](/misˈɑki/) is a G2P engine.";
  const std::string text = "For English, Misaki currently uses 49 total phonemes. Of these, 41 are shared by both Americans and Brits, 4 are American-only, and 4 are British-only. Disclaimer: Author is an ML researcher, not a linguist, and may have butchered or reappropriated the traditional meaning of some symbols. These symbols are intended as input tokens for neural networks to yield optimal performance.";
  const auto result = engine->phonemize_with_tokens(text);

  std::cout << "Input:    " << text << "\n";
  std::cout << "Phonemes: " << result.phonemes << "\n";
  std::cout << "Number Of Tokens: " << result.tokens.size() << "\n";
  std::cout << "\nTokens:\n";
  for (const auto &token : result.tokens) {
    std::cout << "- text=\"" << token.text << "\""
              << " tag=\"" << token.tag << "\""
              << " whitespace=\"" << token.whitespace << "\""
              << " phonemes=\"" << (token.phonemes ? *token.phonemes : std::string{"<null>"}) << "\"\n";
  }

  return 0;
}
