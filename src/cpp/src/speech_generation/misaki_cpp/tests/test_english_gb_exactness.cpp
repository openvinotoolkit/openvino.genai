#include "misaki/g2p.hpp"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using nlohmann::json;

struct EnglishGBGoldenCase {
  std::string text;
  std::string phonemes;
};

static std::vector<EnglishGBGoldenCase> load_english_gb_cases(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.good());

  std::vector<EnglishGBGoldenCase> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto j = json::parse(line);
    REQUIRE(j.at("lang").get<std::string>() == "en");
    REQUIRE(j.at("variant").get<std::string>() == "en-gb");
    out.push_back({
        j.at("text").get<std::string>(),
        j.at("phonemes").get<std::string>(),
    });
  }
  return out;
}

TEST_CASE("English (GB) C++ backend matches English GB golden corpus", "[english][gb][exactness]") {
#ifndef MISAKI_ENGLISH_GB_GOLDEN_CASES_PATH
#define MISAKI_ENGLISH_GB_GOLDEN_CASES_PATH "cpp/tests/english_gb_golden_cases.jsonl"
#endif

  auto cases = load_english_gb_cases(MISAKI_ENGLISH_GB_GOLDEN_CASES_PATH);
  REQUIRE(!cases.empty());

  auto engine = misaki::make_engine("en", "en-gb");
  for (const auto &c : cases) {
    SECTION(c.text) {
      const auto got = engine->phonemize(c.text);
      REQUIRE(got == c.phonemes);
    }
  }
}
