#include "misaki/g2p.hpp"
#include "misaki/fallbacks.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Espeak engine rejects unsupported variant", "[espeak][factory]") {
  REQUIRE_THROWS(misaki::make_engine("espeak", "ja"));
}

TEST_CASE("Espeak engine reports unknown marker when backend unavailable", "[espeak][availability]") {
  misaki::EspeakG2P backend("es", "", "definitely_missing_espeak_ng_library");
  REQUIRE_FALSE(backend.backend_available());
  REQUIRE(backend.backend_error().has_value());

  const auto maybe_ps = backend.phonemize("hola");
  REQUIRE_FALSE(maybe_ps.has_value());
}

TEST_CASE("Espeak engine produces non-empty phonemes when backend available", "[espeak][basic]") {
  auto engine = misaki::make_engine("espeak", "es");
  if (!engine->backend_available()) {
    SKIP("espeak-ng backend is unavailable in this environment");
  }

  const auto result = engine->phonemize_with_tokens("hola mundo");

  REQUIRE(result.tokens.empty());
  REQUIRE_FALSE(result.phonemes.empty());
}
