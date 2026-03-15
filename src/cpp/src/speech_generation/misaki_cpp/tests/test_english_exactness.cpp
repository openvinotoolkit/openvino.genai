#include "misaki/g2p.hpp"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using nlohmann::json;

struct EnglishGoldenCase {
  std::string text;
  std::string phonemes;
};

static std::vector<EnglishGoldenCase> load_english_cases(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.good());

  std::vector<EnglishGoldenCase> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto j = json::parse(line);
    REQUIRE(j.at("lang").get<std::string>() == "en");
    REQUIRE(j.at("variant").get<std::string>() == "en-us");
    out.push_back({
        j.at("text").get<std::string>(),
        j.at("phonemes").get<std::string>(),
    });
  }
  return out;
}

TEST_CASE("English C++ backend matches English golden corpus", "[english][exactness]") {
#ifndef MISAKI_ENGLISH_GOLDEN_CASES_PATH
#define MISAKI_ENGLISH_GOLDEN_CASES_PATH "cpp/tests/english_golden_cases.jsonl"
#endif

  auto cases = load_english_cases(MISAKI_ENGLISH_GOLDEN_CASES_PATH);
  REQUIRE(!cases.empty());

  auto engine = misaki::make_engine("en", "en-us");
  for (const auto &c : cases) {
    SECTION(c.text) {
      const auto got = engine->phonemize(c.text);
      REQUIRE(got == c.phonemes);
    }
  }
}
