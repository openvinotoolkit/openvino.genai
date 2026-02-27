#include "misaki/g2p.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("English sentence parity for seeded golden case", "[english][parity]") {
  auto engine = misaki::make_engine("en", "en-us");
  const std::string input = "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models.";
  const std::string expected = "misˈɑki ɪz ɐ ʤˈitəpˈi ˈɛnʤən dəzˈInd fɔɹ kˈOkəɹO mˈɑdᵊlz.";
  REQUIRE(engine->phonemize(input) == expected);
}

TEST_CASE("English API returns phonemes with tokens", "[english][api][tokens]") {
  auto engine = misaki::make_engine("en", "en-us");
  const auto result = engine->phonemize_with_tokens("[Hello](/həloʊ/) world.");

  REQUIRE(result.phonemes == "həloʊ wˈɜɹld.");
  REQUIRE(result.tokens.size() == 3);
  REQUIRE(result.tokens[0].text == "Hello");
  REQUIRE(result.tokens[0].tag == "WORD");
  REQUIRE(result.tokens[0].phonemes.has_value());
  REQUIRE(*result.tokens[0].phonemes == "həloʊ");
  REQUIRE(result.tokens[0].whitespace == " ");
  REQUIRE_FALSE(result.tokens[0].start_ts.has_value());
  REQUIRE_FALSE(result.tokens[0].end_ts.has_value());
  REQUIRE(result.tokens[0]._.has_value());
  REQUIRE(result.tokens[0]._->is_head.has_value());
  REQUIRE(*result.tokens[0]._->is_head == true);
  REQUIRE(result.tokens[0]._->prespace.has_value());
  REQUIRE(*result.tokens[0]._->prespace == true);
  REQUIRE(result.tokens[0]._->num_flags.empty());
  REQUIRE(result.tokens[0]._->rating.has_value());
  REQUIRE(*result.tokens[0]._->rating == 5);
  REQUIRE(result.tokens[1].text == "world");
  REQUIRE(result.tokens[1].tag == "WORD");
  REQUIRE(result.tokens[1].phonemes.has_value());
  REQUIRE(*result.tokens[1].phonemes == "wˈɜɹld");
  REQUIRE(result.tokens[1].whitespace.empty());
  REQUIRE(result.tokens[2].text == ".");
  REQUIRE(result.tokens[2].tag == "PUNCT");
  REQUIRE(result.tokens[2].phonemes.has_value());
  REQUIRE(*result.tokens[2].phonemes == ".");
}

TEST_CASE("English inline feature directives populate token metadata", "[english][api][tokens][features]") {
  auto engine = misaki::make_engine("en", "en-us");

  const auto stress_result = engine->phonemize_with_tokens("[AIDS](-0.5)");
  REQUIRE(stress_result.tokens.size() == 1);
  REQUIRE(stress_result.tokens[0].phonemes.has_value());
  REQUIRE(*stress_result.tokens[0].phonemes == "ˌAdz");
  REQUIRE(stress_result.tokens[0]._.has_value());
  REQUIRE(stress_result.tokens[0]._->stress.has_value());
  REQUIRE(*stress_result.tokens[0]._->stress == -0.5);

  const auto demote_result = engine->phonemize_with_tokens("[AIDS](0)");
  REQUIRE(demote_result.tokens.size() == 1);
  REQUIRE(demote_result.tokens[0].phonemes.has_value());
  REQUIRE(*demote_result.tokens[0].phonemes == "ˌAdz");

  const auto strip_result = engine->phonemize_with_tokens("[AIDS](-2)");
  REQUIRE(strip_result.tokens.size() == 1);
  REQUIRE(strip_result.tokens[0].phonemes.has_value());
  REQUIRE(*strip_result.tokens[0].phonemes == "Adz");

  const auto flag_result = engine->phonemize_with_tokens("[123](#x#)");
  REQUIRE(flag_result.tokens.size() == 1);
  REQUIRE(flag_result.tokens[0]._.has_value());
  REQUIRE(flag_result.tokens[0]._->num_flags == "x");

  const auto alias_result = engine->phonemize_with_tokens("[placeholder]([to])");
  REQUIRE(alias_result.tokens.size() == 1);
  REQUIRE(alias_result.tokens[0].phonemes.has_value());
  REQUIRE(*alias_result.tokens[0].phonemes == "tu");
  REQUIRE(alias_result.tokens[0]._.has_value());
  REQUIRE(alias_result.tokens[0]._->alias.has_value());
  REQUIRE(*alias_result.tokens[0]._->alias == "to");
  REQUIRE(alias_result.tokens[0]._->rating.has_value());
  REQUIRE(*alias_result.tokens[0]._->rating == 5);
}

TEST_CASE("English inline pronunciation links are respected", "[english][links]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("[Hello](/həloʊ/).") == "həloʊ.");
  REQUIRE(engine->phonemize("[A](/ə/) [B](/bi/)") == "ə bi");
}

TEST_CASE("English lexicon lookup supports common entries", "[english][lexicon]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("AIDS") == "ˈAdz");
  REQUIRE(engine->phonemize("ABTA") == "ˈæbtə");
  REQUIRE(engine->phonemize("A-frame") == "ˈAfɹˌAm");
  REQUIRE(engine->phonemize("you-all") == "ju—ˈɔl");
}

TEST_CASE("English unmapped token uses unknown marker", "[english][errors]") {
  auto engine = misaki::make_engine("en", "en-us");
  const auto result = engine->phonemize_with_tokens("zzzznotawordzzzz");
  REQUIRE(result.phonemes == "❓");
  REQUIRE(result.tokens.size() == 1);
  REQUIRE_FALSE(result.tokens[0].phonemes.has_value());
  REQUIRE(result.tokens[0]._.has_value());
  REQUIRE_FALSE(result.tokens[0]._->rating.has_value());
}

TEST_CASE("English unknown marker is configurable", "[english][errors][unk]") {
  auto engine = misaki::make_engine("en", "en-us");
  engine->set_unknown_token("<UNK>");
  REQUIRE(engine->phonemize("zzzznotawordzzzz") == "<UNK>");
}

TEST_CASE("English lexicon data root is runtime configurable", "[english][data-root]") {
  std::vector<std::filesystem::path> candidates = {
      std::filesystem::current_path() / "data",
      std::filesystem::current_path() / ".." / "data",
      std::filesystem::current_path() / ".." / ".." / "data",
  };

  std::optional<std::filesystem::path> resolved;
  for (const auto &candidate : candidates) {
    if (std::filesystem::exists(candidate / "us_gold.json") &&
        std::filesystem::exists(candidate / "us_silver.json") &&
        std::filesystem::exists(candidate / "gb_gold.json") &&
        std::filesystem::exists(candidate / "gb_silver.json")) {
      resolved = std::filesystem::weakly_canonical(candidate);
      break;
    }
  }

  REQUIRE(resolved.has_value());

  misaki::set_english_lexicon_data_root(resolved->string());
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("tomato") == "təmˈATO");
  misaki::clear_english_lexicon_data_root();
}

TEST_CASE("English fallback hook resolves unknown token", "[english][fallback]") {
  auto engine = misaki::make_engine("en", "en-us");
  engine->set_fallback_hook([](const misaki::MToken &token) -> std::optional<std::string> {
    if (token.text == "zzzznotawordzzzz") {
      return std::string{"fˈɔlbæk"};
    }
    return std::nullopt;
  });

  const auto result = engine->phonemize_with_tokens("zzzznotawordzzzz");
  REQUIRE(result.phonemes == "fˈɔlbæk");
  REQUIRE(result.tokens.size() == 1);
  REQUIRE(result.tokens[0].phonemes.has_value());
  REQUIRE(*result.tokens[0].phonemes == "fˈɔlbæk");
  REQUIRE(result.tokens[0]._.has_value());
  REQUIRE(result.tokens[0]._->rating.has_value());
  REQUIRE(*result.tokens[0]._->rating == 1);
}

TEST_CASE("English en-gb variant produces british lexicon outputs", "[english][gb]") {
  auto engine = misaki::make_engine("en", "en-gb");
  REQUIRE(engine->phonemize("tomato for models you-all.") == "təmˈɑːtQ fɔː mˈɒdᵊlz juː—ˈɔːl.");
}

TEST_CASE("English context-sensitive function words", "[english][context]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("the [apple](/ˈæpəl/)") == "ði ˈæpəl");
  REQUIRE(engine->phonemize("the [model](/mˈɑdəl/)") == "ðə mˈɑdəl");
  REQUIRE(engine->phonemize("to [AIDS](/ˈAdz/)") == "tʊ ˈAdz");
  REQUIRE(engine->phonemize("to [models](/mˈɑdᵊlz/)") == "tə mˈɑdᵊlz");
  REQUIRE(engine->phonemize("a [model](/mˈɑdəl/)") == "ɐ mˈɑdəl");
  REQUIRE(engine->phonemize("A.") == "ˈA.");
}

TEST_CASE("English morphology fallback for suffixes", "[english][morphology]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("walks") == "wˈɔks");
  REQUIRE(engine->phonemize("banked") == "bˈæŋkt");
  REQUIRE(engine->phonemize("played") == "plˈAd");
  REQUIRE(engine->phonemize("tests") == "tˈɛsts");
}

TEST_CASE("English numeric and symbol expansion parity", "[english][numbers][symbols]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("2 + 3") == "tˈu plˈʌs θɹˈi");
  REQUIRE(engine->phonemize("50%") == "fˈɪfti pəɹsˈɛnt");
  REQUIRE(engine->phonemize("@ home") == "æt hˈOm");
  REQUIRE(engine->phonemize("$12.50") == "twˈɛlv dˈɑləɹz ænd fˈɪfti sˈɛnts");
  REQUIRE(engine->phonemize("$1") == "wˈʌn dˈɑləɹ");
  REQUIRE(engine->phonemize("3.14") == "θɹˈi pYnt wˈʌn fˈɔɹ");
}

TEST_CASE("English ordinal and decimal edge parity", "[english][numbers][ordinals]") {
  auto engine = misaki::make_engine("en", "en-us");
  REQUIRE(engine->phonemize("1st") == "fˈɜɹst");
  REQUIRE(engine->phonemize("21st") == "twˈɛnti fˈɜɹst");
  REQUIRE(engine->phonemize("101st") == "wˈʌn hˈʌndɹəd fˈɜɹst");
  REQUIRE(engine->phonemize(".50") == "pYnt fˈIv zˈɪɹO");
  REQUIRE(engine->phonemize("0.50") == "zˈɪɹO pYnt fˈIv");
}

TEST_CASE("English year and numeric-suffix parity", "[english][numbers][years]") {
  auto us = misaki::make_engine("en", "en-us");
  REQUIRE(us->phonemize("1905") == "nˌIntˈin ˈO fˈIv");
  REQUIRE(us->phonemize("2024") == "twˈɛnti twˈɛnti fˈɔɹ");
  REQUIRE(us->phonemize("1990s") == "nˌIntˈin nˈIndiz");
  REQUIRE(us->phonemize("1990's") == "nˌIntˈin nˈIndiz");
  REQUIRE(us->phonemize("2000s") == "tˈu θˈWzᵊndz");
  REQUIRE(us->phonemize("12ed") == "twˈɛlvd");
  REQUIRE(us->phonemize("12ing") == "twˈɛlvɪŋ");
  REQUIRE(us->phonemize("$0.00") == "zˈɪɹO dˈɑləɹz");
  REQUIRE(us->phonemize("$0.01") == "wˈʌn sˈɛnt");
  REQUIRE(us->phonemize("$1.01") == "wˈʌn dˈɑləɹ ænd wˈʌn sˈɛnt");

  auto gb = misaki::make_engine("en", "en-gb");
  REQUIRE(gb->phonemize("1905") == "nˌIntˈiːn ˈQ fˈIv");
  REQUIRE(gb->phonemize("2024") == "twˈɛnti twˈɛnti fˈɔː");
  REQUIRE(gb->phonemize("1990s") == "nˌIntˈiːn nˈIntiz");
  REQUIRE(gb->phonemize("1990's") == "nˌIntˈiːn nˈIntiz");
  REQUIRE(gb->phonemize("2000s") == "tˈuː θˈWzᵊndz");
  REQUIRE(gb->phonemize("$0.00") == "zˈɪəɹQ dˈɒləz");
  REQUIRE(gb->phonemize("$0.01") == "wˈʌn sˈɛnt");
  REQUIRE(gb->phonemize("$1.01") == "wˈʌn dˈɒlə and wˈʌn sˈɛnt");
}

TEST_CASE("English context phrase parity", "[english][context][phrases]") {
  auto us = misaki::make_engine("en", "en-us");
  REQUIRE(us->phonemize("used to") == "jˈust tu");
  REQUIRE(us->phonemize("I used to go") == "ˌI jˈust tə ɡˌO");
  REQUIRE(us->phonemize("am") == "æm");
  REQUIRE(us->phonemize("I am here") == "ˌI ɐm hˈɪɹ");
  REQUIRE(us->phonemize("vs.") == "vˈɜɹsəs");
  REQUIRE(us->phonemize("A vs. B") == "ɐ vˈiz bˈi");
  REQUIRE(us->phonemize("in") == "ˈɪn");
  REQUIRE(us->phonemize("in town") == "ɪn tˈWn");
  REQUIRE(us->phonemize("by") == "bˈI");
  REQUIRE(us->phonemize("by far") == "bI fˈɑɹ");

  auto gb = misaki::make_engine("en", "en-gb");
  REQUIRE(gb->phonemize("used to") == "jˈuːst tuː");
  REQUIRE(gb->phonemize("I used to go") == "ˌI jˈuːst tə ɡˌQ");
  REQUIRE(gb->phonemize("am") == "am");
  REQUIRE(gb->phonemize("I am here") == "ˌI ɐm hˈɪə");
  REQUIRE(gb->phonemize("vs.") == "vˈɜːsəs");
  REQUIRE(gb->phonemize("A vs. B") == "ɐ vˈiːz bˈiː");
  REQUIRE(gb->phonemize("in") == "ˈɪn");
  REQUIRE(gb->phonemize("in town") == "ɪn tˈWn");
  REQUIRE(gb->phonemize("by") == "bˈI");
  REQUIRE(gb->phonemize("by far") == "bI fˈɑː");
}

TEST_CASE("English hyphen compound parity", "[english][hyphen]") {
  auto us = misaki::make_engine("en", "en-us");
  auto gb = misaki::make_engine("en", "en-gb");

  REQUIRE(us->phonemize("bye-bye") == "bˈI—bˈI");
  REQUIRE(gb->phonemize("bye-bye") == "bˈI—bˈI");
}

TEST_CASE("English lexicon dict POS hint parity", "[english][pos][lexicon]") {
  auto us = misaki::make_engine("en", "en-us");
  REQUIRE(us->phonemize("conduct") == "kˈɑndˌʌkt");
  REQUIRE(us->phonemize("to conduct") == "tə kəndˈʌkt");
  REQUIRE(us->phonemize("I conduct") == "ˌI kəndˈʌkt");
  REQUIRE(us->phonemize("will conduct") == "wɪl kəndˈʌkt");
  REQUIRE(us->phonemize("contract") == "kˈɑntɹˌækt");
  REQUIRE(us->phonemize("to contract") == "tə kəntɹˈækt");
  REQUIRE(us->phonemize("I contract") == "ˌI kəntɹˈækt");
  REQUIRE(us->phonemize("to conflict") == "tə kənflˈɪkt");
  REQUIRE(us->phonemize("to record") == "tə ɹəkˈɔɹd");

  auto gb = misaki::make_engine("en", "en-gb");
  REQUIRE(gb->phonemize("conduct") == "kˈɒndʌkt");
  REQUIRE(gb->phonemize("to conduct") == "tə kəndˈʌkt");
  REQUIRE(gb->phonemize("I conduct") == "ˌI kəndˈʌkt");
  REQUIRE(gb->phonemize("will conduct") == "wɪl kəndˈʌkt");
  REQUIRE(gb->phonemize("contract") == "kˈɒntɹakt");
  REQUIRE(gb->phonemize("to contract") == "tə kəntɹˈakt");
  REQUIRE(gb->phonemize("I contract") == "ˌI kəntɹˈakt");
  REQUIRE(gb->phonemize("to conflict") == "tə kənflˈɪkt");
  REQUIRE(gb->phonemize("to record") == "tə ɹɪkˈɔːd");
}

TEST_CASE("English capitalization stress and hyphen compounds", "[english][caps][hyphen]") {
  auto us = misaki::make_engine("en", "en-us");
  REQUIRE(us->phonemize("For") == "fˌɔɹ");
  REQUIRE(us->phonemize("American-only") == "əmˈɛɹəkᵊnˌOnli");
  REQUIRE(us->phonemize("British-only") == "bɹˈɪTɪʃˌOnli");
}
