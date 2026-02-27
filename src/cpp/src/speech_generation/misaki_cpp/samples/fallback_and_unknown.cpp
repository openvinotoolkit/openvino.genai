#include "misaki/g2p.hpp"

#include <iostream>
#include <optional>
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

  engine->set_unknown_token("<UNK>");
  engine->set_fallback_hook([](const misaki::MToken &token) -> std::optional<std::string> {
    if (token.text == "outofdictionary") {
      return std::string{"ˌWtəvdˈɪkʃəˌnɛɹi"};
    }
    return std::nullopt;
  });

  const std::string text = "Okay I am adding an outofdictionary word";
  const auto result = engine->phonemize_with_tokens(text);

  std::cout << "Input:    " << text << "\n";
  std::cout << "Phonemes: " << result.phonemes << "\n";

  std::cout << "\nFallback/unknown view:\n";
  for (const auto &token : result.tokens) {
    const auto value = token.phonemes ? *token.phonemes : std::string{"<null>"};
    std::cout << "- " << token.text << " => " << value;
    if (token._ && token._->rating) {
      std::cout << " (rating=" << *token._->rating << ")";
    }
    std::cout << "\n";
  }

  return 0;
}
