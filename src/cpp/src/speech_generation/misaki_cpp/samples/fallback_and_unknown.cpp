#include "misaki/g2p.hpp"
#include "misaki/fallbacks.hpp"

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

void print_utf8_console_diagnostics() {
#ifdef _WIN32
  const UINT input_cp = GetConsoleCP();
  const UINT output_cp = GetConsoleOutputCP();

  std::cout << "[Info] Console CP (in/out): " << input_cp << "/" << output_cp << "\n";
  if (input_cp != CP_UTF8 || output_cp != CP_UTF8) {
    std::cout << "[Info] UTF-8 may be misconfigured. In a new terminal run: chcp 65001\n";
  }
  std::cout << "[Info] If IPA glyphs still look garbled, use a Unicode-capable terminal font.\n";
#endif
}
} // namespace

int main() {
  configure_utf8_console();
  print_utf8_console_diagnostics();

  auto engine = misaki::make_engine("en", "en-us");
  misaki::EspeakFallback espeak_fallback(/*british=*/false);

  if (!espeak_fallback.backend_available()) {
    if (const auto err = espeak_fallback.backend_error(); err.has_value()) {
      std::cerr << "[Info] EspeakFallback unavailable: " << *err << "\n";
    }
  }

  engine->set_unknown_token("<UNK>");
  engine->set_fallback_hook([espeak_fallback](const misaki::MToken &token) -> std::optional<std::string> {
    if (token.text == "outofdictionary") {
      return std::string{"ˌWtəvdˈɪkʃəˌnɛɹi"};
    }
    return espeak_fallback(token);
  });

  //const std::string text = "Okay I am adding an outofdictionary word";
  const std::string text = "I stepped off the NovaLiner-3 at Aeroluna Station and met Dr. Vellorin Quade from the Lumenfield R&D Lab (badge ID: QD-17B). He handed me a twinklepack of windleberries";
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
