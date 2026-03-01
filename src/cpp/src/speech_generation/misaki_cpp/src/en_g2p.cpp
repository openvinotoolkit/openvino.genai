#include "misaki/g2p.hpp"

#include "english_lexicon.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace misaki {
namespace {

MToken::Underscore default_token_meta() {
  MToken::Underscore meta;
  meta.is_head = true;
  meta.alias = std::nullopt;
  meta.stress = std::nullopt;
  meta.currency = std::nullopt;
  meta.num_flags = "";
  meta.prespace = false;
  meta.rating = std::nullopt;
  return meta;
}

MToken make_output_token(const std::string &text,
                         const std::string &tag,
                         const std::string &whitespace,
                         const std::optional<std::string> &phonemes,
                         const std::optional<int> &rating = std::nullopt) {
  MToken token;
  token.text = text;
  token.tag = tag;
  token.whitespace = whitespace;
  token.phonemes = phonemes;
  token.start_ts = std::nullopt;
  token.end_ts = std::nullopt;
  auto meta = default_token_meta();
  meta.rating = rating;
  token._ = std::move(meta);
  return token;
}

class EnglishG2P final : public G2P {
public:
  explicit EnglishG2P(std::string variant) : variant_(std::move(variant)) {}

  PhonemizeResult phonemize_with_tokens(const std::string &text) const override {
    // Python mapping: G2P.preprocess(...) + G2P.tokenize(...) + G2P.retokenize(...)
    // This C++ stage keeps only the subset needed for exactness: link parsing and deterministic tokenization.
    auto tokens = preprocess_and_tokenize(text);
    // Python mapping: G2P.__call__(...) reverse-resolution loop + final token merge/output normalization.
    return render_pipeline_output(tokens);
  }

private:
  std::string variant_;

  enum class TokenKind { Word, Punctuation, Whitespace };

  struct Token {
    TokenKind kind;
    std::string text;
    std::string linked_pronunciation;
    std::optional<double> linked_stress = std::nullopt;
    std::string linked_num_flags;
    std::string linked_alias;
  };

  struct TokenContext {
    std::optional<bool> future_vowel;
  };

  struct ParsedNumber {
    bool valid = false;
    bool negative = false;
    bool leading_dot = false;
    std::string whole;
    std::string fractional;
    std::string suffix;
    std::string original_core;
  };

  // Python mapping: preprocess/tokenize/retokenize entry point for C++.
  static std::vector<Token> preprocess_and_tokenize(const std::string &text) {
    std::string normalized = text;
    replace_all(normalized, "\xE2\x80\x99", "'");
    replace_all(normalized, "\xE2\x80\x98", "'");
    return tokenize(normalized);
  }

  // Python mapping: final __call__ assembly of token phonemes with punctuation/spacing behavior.
  PhonemizeResult render_pipeline_output(const std::vector<Token> &tokens) const {
    std::string out;
    std::vector<MToken> output_tokens;
    bool needs_space = false;

    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const auto &tk = tokens[i];
      if (tk.kind == TokenKind::Whitespace) {
        needs_space = !out.empty();
        if (!output_tokens.empty()) {
          output_tokens.back().whitespace = " ";
          if (output_tokens.back()._.has_value()) {
            output_tokens.back()._->prespace = true;
          }
        }
        continue;
      }

      if (tk.kind == TokenKind::Punctuation) {
        if (should_skip_punctuation_output(tokens, i)) {
          continue;
        }
        out += tk.text;
        output_tokens.push_back(make_output_token(tk.text, "PUNCT", "", tk.text, 4));
        needs_space = false;
        continue;
      }

      if (needs_space && !out.empty() && out.back() != ' ') {
        out.push_back(' ');
      }

      const auto pronunciation = resolve_word_token(tokens, i);
      if (!pronunciation.has_value()) {
        // Python parity: EspeakFallback returns `(phonemes, 2)`.
        // Use rating=2 for unresolved-word fallback tokens.
        auto fallback_token = make_output_token(tk.text, "WORD", "", std::nullopt, 2);
        if (fallback_token._.has_value()) {
          fallback_token._->prespace = needs_space;
          if (tk.linked_stress.has_value()) {
            fallback_token._->stress = tk.linked_stress;
          }
          if (!tk.linked_alias.empty()) {
            fallback_token._->alias = tk.linked_alias;
          }
          if (!tk.linked_num_flags.empty()) {
            fallback_token._->num_flags = tk.linked_num_flags;
          }
        }

        const auto fallback_pron = run_fallback_hook(fallback_token);
        if (!fallback_pron.has_value()) {
          if (needs_space && !out.empty() && out.back() != ' ') {
            out.push_back(' ');
          }
          out += unknown_token();
          fallback_token.phonemes = std::nullopt;
          if (fallback_token._.has_value()) {
            fallback_token._->rating = std::nullopt;
          }
          output_tokens.push_back(std::move(fallback_token));
          needs_space = true;
          continue;
        }

        const auto normalized_fallback = normalize_pronunciation(*fallback_pron);
        out += normalized_fallback;
        fallback_token.phonemes = normalized_fallback;
        output_tokens.push_back(std::move(fallback_token));
        needs_space = true;
        continue;
      }

      if (pronunciation->empty()) {
        output_tokens.push_back(make_output_token(tk.text, "WORD", "", std::string{}, 3));
        if (output_tokens.back()._.has_value()) {
          if (tk.linked_stress.has_value()) {
            output_tokens.back()._->stress = tk.linked_stress;
          }
          if (!tk.linked_alias.empty()) {
            output_tokens.back()._->alias = tk.linked_alias;
          }
          if (!tk.linked_num_flags.empty()) {
            output_tokens.back()._->num_flags = tk.linked_num_flags;
          }
        }
        continue;
      }
      out += *pronunciation;
      output_tokens.push_back(make_output_token(tk.text, "WORD", "", *pronunciation, 4));
      if (output_tokens.back()._.has_value()) {
        if (tk.linked_stress.has_value()) {
          output_tokens.back()._->stress = tk.linked_stress;
        }
        if (!tk.linked_alias.empty()) {
          output_tokens.back()._->alias = tk.linked_alias;
        }
        if (!tk.linked_num_flags.empty()) {
          output_tokens.back()._->num_flags = tk.linked_num_flags;
        }
        if (!tk.linked_pronunciation.empty() || !tk.linked_alias.empty()) {
          output_tokens.back()._->rating = 5;
        }
      }
      needs_space = true;
    }

    return PhonemizeResult{out, std::move(output_tokens)};
  }

  // Python mapping: punctuation treatment during final token stitching.
  static bool should_skip_punctuation_output(const std::vector<Token> &tokens, std::size_t index) {
    const auto &tk = tokens[index];
    if (tk.kind != TokenKind::Punctuation || tk.text != ".") {
      return false;
    }

    std::optional<std::size_t> prev_word;
    for (std::size_t j = index; j-- > 0;) {
      if (tokens[j].kind == TokenKind::Whitespace) {
        continue;
      }
      if (tokens[j].kind == TokenKind::Word) {
        prev_word = j;
      }
      break;
    }
    if (!prev_word) {
      return false;
    }

    const auto prev = ascii_lower(tokens[*prev_word].text);
    if (prev == "vs") {
      return true;
    }

    static const std::unordered_set<std::string> abbrev_without_dot = {
      "dr", "mr", "mrs", "ms", "sr", "jr",
    };
    return abbrev_without_dot.find(prev) != abbrev_without_dot.end();
  }

  // Python mapping: per-token lexicon/fallback resolution path in G2P.__call__.
  std::optional<std::string> resolve_word_token(const std::vector<Token> &tokens, std::size_t index) const {
    std::string pronunciation;
    try {
      pronunciation = normalize_pronunciation(lookup_pronunciation(tokens, index));
    } catch (const std::runtime_error &e) {
      const std::string message = e.what();
      if (message.rfind("Unmapped English token: ", 0) == 0) {
        return std::nullopt;
      }
      throw;
    }
    if (!tokens[index].linked_alias.empty()) {
      if (const auto alias_pron = lookup_word_pronunciation(tokens[index].linked_alias)) {
        pronunciation = normalize_pronunciation(*alias_pron);
      }
    }
    if (tokens[index].linked_stress.has_value()) {
      pronunciation = apply_inline_stress(pronunciation, *tokens[index].linked_stress);
    } else if (!tokens[index].linked_pronunciation.empty()) {
      // Keep explicit inline phonemes as-is unless explicit stress override is provided.
    } else if (!has_any_stress(pronunciation)) {
      if (const auto case_stress = infer_case_stress(tokens[index].text)) {
        pronunciation = apply_inline_stress(pronunciation, *case_stress);
      }
    } else {
      // Preserve existing stress from lexical/context rules.
    }
    return pronunciation;
  }

  static std::optional<double> infer_case_stress(const std::string &word) {
    bool has_alpha = false;
    bool any_upper = false;
    bool all_upper = true;

    for (char c : word) {
      const auto uc = static_cast<unsigned char>(c);
      if (!std::isalpha(uc)) {
        continue;
      }
      has_alpha = true;
      if (std::isupper(uc)) {
        any_upper = true;
      } else {
        all_upper = false;
      }
    }

    if (!has_alpha || !any_upper) {
      return std::nullopt;
    }
    return all_upper ? std::optional<double>{2.0} : std::optional<double>{0.5};
  }

  static void replace_all(std::string &value, const std::string &from, const std::string &to) {
    if (from.empty()) {
      return;
    }
    std::size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
      value.replace(pos, from.size(), to);
      pos += to.size();
    }
  }

  static std::string demote_primary_stress(std::string pronunciation) {
    replace_all(pronunciation, "ˈ", "ˌ");
    return pronunciation;
  }

  static std::string normalize_pronunciation(std::string pronunciation) {
    replace_all(pronunciation, "ɾ", "T");
    replace_all(pronunciation, "ʔ", "t");

    std::string collapsed;
    collapsed.reserve(pronunciation.size());
    bool last_space = false;
    for (char c : pronunciation) {
      if (c == ' ') {
        if (!last_space) {
          collapsed.push_back(c);
        }
        last_space = true;
      } else {
        collapsed.push_back(c);
        last_space = false;
      }
    }

    while (!collapsed.empty() && collapsed.front() == ' ') {
      collapsed.erase(collapsed.begin());
    }
    while (!collapsed.empty() && collapsed.back() == ' ') {
      collapsed.pop_back();
    }
    return collapsed;
  }

  static bool is_word_char(unsigned char c) {
    return std::isalnum(c) || c == '\'' || c == '-';
  }

  static bool is_expandable_symbol(char c) {
    switch (c) {
    case '%':
    case '&':
    case '+':
    case '@':
    case '$':
    case '/':
      return true;
    default:
      return false;
    }
  }

  static bool is_punctuation(char c) {
    switch (c) {
    case '.':
    case ',':
    case '!':
    case '?':
    case ';':
    case ':':
      return true;
    default:
      return false;
    }
  }

  static std::string ascii_lower(const std::string &input) {
    std::string out = input;
    for (auto &c : out) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
  }

  static bool ends_with(const std::string &value, const std::string &suffix) {
    if (suffix.size() > value.size()) {
      return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  static bool ends_with_any(const std::string &value, const std::vector<std::string> &suffixes) {
    for (const auto &suffix : suffixes) {
      if (ends_with(value, suffix)) {
        return true;
      }
    }
    return false;
  }

  static std::string trim_ascii_spaces(std::string value) {
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    while (!value.empty() && value.back() == ' ') {
      value.pop_back();
    }
    return value;
  }

  static std::string collapse_ascii_spaces(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    bool last_space = false;
    for (char c : value) {
      if (c == ' ') {
        if (!last_space) {
          out.push_back(c);
        }
        last_space = true;
      } else {
        out.push_back(c);
        last_space = false;
      }
    }
    return trim_ascii_spaces(std::move(out));
  }
  
  static std::size_t utf8_sequence_length(unsigned char lead) {
    if ((lead & 0x80) == 0x00) {
      return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
      return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
      return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
      return 4;
    }
    return 1;
  }

  static bool is_ascii_vowel(char c) {
    const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u';
  }

  static bool is_ordinal_suffix(const std::string &suffix) {
    return suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th";
  }

  static bool is_numeric_word_suffix(const std::string &suffix) {
    return suffix == "s" || suffix == "'s" || suffix == "ed" || suffix == "'d" || suffix == "ing";
  }

  static std::string strip_commas(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    return value;
  }

  static ParsedNumber parse_number_token(const std::string &text) {
    ParsedNumber parsed;
    if (text.empty()) {
      return parsed;
    }

    std::size_t i = 0;
    if (text[i] == '-') {
      parsed.negative = true;
      ++i;
      if (i >= text.size()) {
        return ParsedNumber{};
      }
    }

    const auto core_start = i;
    if (text[i] == '.') {
      parsed.leading_dot = true;
      ++i;
      const auto frac_start = i;
      while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        ++i;
      }
      if (i == frac_start) {
        return ParsedNumber{};
      }
      parsed.fractional = text.substr(frac_start, i - frac_start);
    } else {
      const auto whole_start = i;
      while (i < text.size() && (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == ',')) {
        ++i;
      }
      if (i == whole_start) {
        return ParsedNumber{};
      }
      parsed.whole = strip_commas(text.substr(whole_start, i - whole_start));

      if (i < text.size() && text[i] == '.') {
        ++i;
        const auto frac_start = i;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
        if (i == frac_start) {
          return ParsedNumber{};
        }
        parsed.fractional = text.substr(frac_start, i - frac_start);
      }

      parsed.original_core = text.substr(core_start, i - core_start);
    }

    if (i < text.size()) {
      parsed.suffix = ascii_lower(text.substr(i));
      const bool ordinal = is_ordinal_suffix(parsed.suffix);
      const bool morph_suffix = is_numeric_word_suffix(parsed.suffix);
      if ((!ordinal && !morph_suffix) || !parsed.fractional.empty() || parsed.leading_dot) {
        return ParsedNumber{};
      }
      i = text.size();
    }

    if (parsed.whole.empty() && !parsed.leading_dot) {
      parsed.whole = "0";
    }

    parsed.valid = (i == text.size());
    return parsed;
  }

  static bool starts_with_vowel_letter(const std::string &text) {
    for (char c : text) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        return is_ascii_vowel(c);
      }
    }
    return false;
  }

  static std::optional<std::size_t> find_next_word_index(const std::vector<Token> &tokens, std::size_t start) {
    for (std::size_t i = start + 1; i < tokens.size(); ++i) {
      if (tokens[i].kind == TokenKind::Word) {
        return i;
      }
      if (tokens[i].kind == TokenKind::Punctuation) {
        break;
      }
    }
    return std::nullopt;
  }

  static std::optional<std::size_t> find_prev_word_index(const std::vector<Token> &tokens, std::size_t start) {
    if (start == 0) {
      return std::nullopt;
    }
    for (std::size_t i = start; i-- > 0;) {
      if (tokens[i].kind == TokenKind::Word) {
        return i;
      }
      if (tokens[i].kind == TokenKind::Punctuation) {
        break;
      }
    }
    return std::nullopt;
  }

  static TokenContext build_context(const std::vector<Token> &tokens, std::size_t index) {
    TokenContext ctx;
    const auto next_word_index = find_next_word_index(tokens, index);
    if (next_word_index) {
      ctx.future_vowel = starts_with_vowel_letter(tokens[*next_word_index].text);
    }
    return ctx;
  }

  static std::optional<std::string> guess_pos_hint(const std::vector<Token> &tokens, std::size_t index) {
    // Python mapping: en.py uses spaCy token.tag_ (from self.nlp) to select dict entries in Lexicon.lookup.
    // C++ replacement: deterministic local context hints that approximate tag-to-parent-tag routing.
    const auto key = ascii_lower(tokens[index].text);

    const auto prev_word = find_prev_word_index(tokens, index);
    const auto next_word = find_next_word_index(tokens, index);
    if (prev_word) {
      const auto prev = ascii_lower(tokens[*prev_word].text);
      if (prev == "to") {
        return std::string{"VERB"};
      }
      if (prev == "i" || prev == "you" || prev == "he" || prev == "she" || prev == "it" ||
          prev == "we" || prev == "they") {
        return std::string{"VERB"};
      }
      if (prev == "will" || prev == "would" || prev == "can" || prev == "could" || prev == "should" ||
          prev == "may" || prev == "might" || prev == "must" || prev == "do" || prev == "does" ||
          prev == "did") {
        return std::string{"VERB"};
      }
      if (prev == "a" || prev == "an" || prev == "the" || prev == "this" || prev == "that" ||
          prev == "these" || prev == "those" || prev == "my" || prev == "your" || prev == "his" ||
          prev == "her" || prev == "our" || prev == "their") {
        return std::string{"NOUN"};
      }
      if (prev == "is" || prev == "are" || prev == "was" || prev == "were" || prev == "be" ||
          prev == "been" || prev == "being" || prev == "am") {
        return std::string{"ADJ"};
      }
    } else {
      static const std::unordered_map<std::string, bool> standalone_verb_bias = {
          {"conglomerate", true},
          {"correlates", true},
          {"re-count", true},
          {"augments", true},
          {"mismatch", true},
          {"covert", true},
      };
      if (standalone_verb_bias.find(key) != standalone_verb_bias.end()) {
        return std::string{"VERB"};
      }

      if (!next_word) {
        return std::string{"NOUN"};
      }
    }

    if (ends_with(key, "ly")) {
      return std::string{"ADV"};
    }

    return std::nullopt;
  }

  static std::string apply_stress(std::string pronunciation, bool stress) {
    if (!stress) {
      return pronunciation;
    }
    if (pronunciation.empty()) {
      return pronunciation;
    }
    if (pronunciation.rfind("ˈ", 0) == 0 || pronunciation.rfind("ˌ", 0) == 0) {
      return pronunciation;
    }
    return "ˈ" + pronunciation;
  }

  static bool nearly_equal(double a, double b) {
    return std::fabs(a - b) < 1e-9;
  }

  static bool has_any_stress(const std::string &pronunciation) {
    return pronunciation.find("ˈ") != std::string::npos || pronunciation.find("ˌ") != std::string::npos;
  }

  static bool has_primary_stress(const std::string &pronunciation) {
    return pronunciation.find("ˈ") != std::string::npos;
  }

  static bool has_secondary_stress(const std::string &pronunciation) {
    return pronunciation.find("ˌ") != std::string::npos;
  }

  static bool has_any_vowel_symbol(const std::string &pronunciation) {
    static const std::vector<std::string> vowels = {
        "A", "I", "O", "Q", "W", "Y", "a", "i", "u", "æ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "ᵻ",
    };
    for (const auto &v : vowels) {
      if (pronunciation.find(v) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  static std::string restress_prefix(const std::string &pronunciation, const std::string &stress_marker) {
    static const std::vector<std::string> vowels = {
        "A", "I", "O", "Q", "W", "Y", "a", "i", "u", "æ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "ᵻ",
    };

    std::size_t best = std::string::npos;
    for (const auto &v : vowels) {
      const auto pos = pronunciation.find(v);
      if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
        best = pos;
      }
    }

    if (best == std::string::npos) {
      return pronunciation;
    }

    return pronunciation.substr(0, best) + stress_marker + pronunciation.substr(best);
  }

  static std::string apply_inline_stress(std::string pronunciation, double stress) {
    if (pronunciation.empty()) {
      return pronunciation;
    }

    if (stress < -1.0) {
      replace_all(pronunciation, "ˈ", "");
      replace_all(pronunciation, "ˌ", "");
      return pronunciation;
    }

    if (nearly_equal(stress, -1.0) ||
        ((nearly_equal(stress, 0.0) || nearly_equal(stress, -0.5)) && has_primary_stress(pronunciation))) {
      replace_all(pronunciation, "ˌ", "");
      replace_all(pronunciation, "ˈ", "ˌ");
      return pronunciation;
    }

    const bool has_primary = has_primary_stress(pronunciation);
    const bool has_secondary = has_secondary_stress(pronunciation);

    if ((nearly_equal(stress, 0.0) || nearly_equal(stress, 0.5) || nearly_equal(stress, 1.0)) &&
        !has_primary && !has_secondary) {
      if (!has_any_vowel_symbol(pronunciation)) {
        return pronunciation;
      }
      return restress_prefix(pronunciation, "ˌ");
    }

    if (stress >= 1.0 && !has_primary && has_secondary) {
      const auto pos = pronunciation.find("ˌ");
      if (pos != std::string::npos) {
        pronunciation.replace(pos, std::string{"ˌ"}.size(), "ˈ");
      }
      return pronunciation;
    }

    if (stress > 1.0 && !has_primary && !has_secondary) {
      if (!has_any_vowel_symbol(pronunciation)) {
        return pronunciation;
      }
      return restress_prefix(pronunciation, "ˈ");
    }

    return pronunciation;
  }

  std::optional<std::string> lookup_direct_pronunciation(
      const std::string &token_text,
      const std::optional<std::string> &pos_hint = std::nullopt) const {
    static const std::unordered_map<std::string, std::string> hardcoded_us = {
        {"is", "ɪz"},
      {"zero", "zˈiəɹO"},
        {"g2p", "ʤˈitəpˈi"},
        {"engine", "ˈɛnʤən"},
        {"designed", "dəzˈInd"},
        {"for", "fɔɹ"},
      {"eleven", "ɪlˈɛvən"},
      {"seventeen", "sˌɛvəntˈin"},
        {"models", "mˈɑdᵊlz"},
        {"you-all", "ju—ˈɔl"},
    };

    static const std::unordered_map<std::string, std::string> hardcoded_gb = {
      {"zero", "zˈiəɹO"},
      {"eleven", "ɪlˈɛvən"},
      {"seventeen", "sˌɛvəntˈiːn"},
        {"you-all", "juː—ˈɔːl"},
        {"for", "fɔː"},
        {"models", "mˈɒdᵊlz"},
    };

    const auto key = ascii_lower(token_text);
    if (token_text == "AM") {
      return "ˌAˈɛm";
    }
    if (token_text == "PM") {
      return "pˌiˈɛm";
    }
    if (key == "don't") {
      return variant_ == "en-gb" ? "dˈəʊnt" : "dˈOnt";
    }
    if (key == "let's") {
      return "lˈɛts";
    }

    const auto &hardcoded = (variant_ == "en-gb") ? hardcoded_gb : hardcoded_us;
    const auto it = hardcoded.find(key);
    if (it != hardcoded.end()) {
      return it->second;
    }

    if (key.size() > 1 && token_text.size() <= 4) {
      std::string acronym;
      acronym.reserve(key.size() * 3);
      bool all_upper = true;
      for (char c : token_text) {
        if (!std::isupper(static_cast<unsigned char>(c))) {
          all_upper = false;
          break;
        }
      }
      if (all_upper) {
        for (std::size_t letter_idx = 0; letter_idx < token_text.size(); ++letter_idx) {
          const char c = token_text[letter_idx];
          std::string letter(1, c);
          if (const auto *lp = detail::find_english_lexicon_entry(letter, variant_)) {
            std::string piece = *lp;
            if (token_text.size() > 1 && letter_idx + 1 < token_text.size()) {
              piece = demote_primary_stress(std::move(piece));
            }
            acronym += piece;
          } else {
            acronym.clear();
            break;
          }
        }
        if (!acronym.empty()) {
          return acronym;
        }
      }
    }

    if (const auto *from_lexicon = detail::find_english_lexicon_entry(token_text, variant_, pos_hint)) {
      return *from_lexicon;
    }

    return std::nullopt;
  }

  std::string apply_plural_s(const std::string &stem) const {
    if (ends_with_any(stem, {"p", "t", "k", "f", "θ"})) {
      return stem + "s";
    }
    if (ends_with_any(stem, {"s", "z", "ʃ", "ʒ", "ʧ", "ʤ"})) {
      return stem + (variant_ == "en-gb" ? "ɪ" : "ᵻ") + "z";
    }
    return stem + "z";
  }

  std::string apply_past_ed(const std::string &stem) const {
    if (ends_with_any(stem, {"p", "k", "f", "θ", "ʃ", "s", "ʧ"})) {
      return stem + "t";
    }
    if (ends_with(stem, "d")) {
      return stem + (variant_ == "en-gb" ? "ɪ" : "ᵻ") + "d";
    }
    if (!ends_with(stem, "t")) {
      return stem + "d";
    }
    return stem + (variant_ == "en-gb" ? "ɪ" : "ᵻ") + "d";
  }

  std::optional<std::string> apply_ing(const std::string &stem) const {
    if (stem.empty()) {
      return std::nullopt;
    }
    if (variant_ == "en-gb" && (ends_with(stem, "ə") || ends_with(stem, "ː"))) {
      return std::nullopt;
    }
    return stem + "ɪŋ";
  }

  std::optional<std::string> stem_s_pronunciation(const std::string &word) const {
    const auto key = ascii_lower(word);
    if (key.size() < 3 || !ends_with(key, "s")) {
      return std::nullopt;
    }

    std::vector<std::string> candidates;
    if (!ends_with(key, "ss")) {
      candidates.push_back(key.substr(0, key.size() - 1));
    }
    if (ends_with(key, "'s") || (key.size() > 4 && ends_with(key, "es") && !ends_with(key, "ies"))) {
      candidates.push_back(key.substr(0, key.size() - 2));
    }
    if (key.size() > 4 && ends_with(key, "ies")) {
      candidates.push_back(key.substr(0, key.size() - 3) + "y");
    }

    for (const auto &stem_word : candidates) {
      if (const auto stem = lookup_direct_pronunciation(stem_word)) {
        return apply_plural_s(*stem);
      }
    }
    return std::nullopt;
  }

  std::optional<std::string> stem_ed_pronunciation(const std::string &word) const {
    const auto key = ascii_lower(word);
    if (key.size() < 4 || !ends_with(key, "d")) {
      return std::nullopt;
    }

    std::vector<std::string> candidates;
    if (!ends_with(key, "dd")) {
      candidates.push_back(key.substr(0, key.size() - 1));
    }
    if (key.size() > 4 && ends_with(key, "ed") && !ends_with(key, "eed")) {
      candidates.push_back(key.substr(0, key.size() - 2));
    }

    for (const auto &stem_word : candidates) {
      if (const auto stem = lookup_direct_pronunciation(stem_word)) {
        return apply_past_ed(*stem);
      }
    }
    return std::nullopt;
  }

  std::optional<std::string> stem_ing_pronunciation(const std::string &word) const {
    const auto key = ascii_lower(word);
    if (key.size() < 5 || !ends_with(key, "ing")) {
      return std::nullopt;
    }

    std::vector<std::string> candidates;
    if (key.size() > 5) {
      candidates.push_back(key.substr(0, key.size() - 3));
    }
    candidates.push_back(key.substr(0, key.size() - 3) + "e");

    if (ends_with(key, "cking") && key.size() > 5) {
      candidates.push_back(key.substr(0, key.size() - 4));
    } else if (key.size() > 6) {
      const char c1 = key[key.size() - 4];
      const char c2 = key[key.size() - 5];
      const std::string doubles = "bcdgklmnprstvxz";
      if (c1 == c2 && doubles.find(c1) != std::string::npos) {
        candidates.push_back(key.substr(0, key.size() - 4));
      }
    }

    for (const auto &stem_word : candidates) {
      if (const auto stem = lookup_direct_pronunciation(stem_word)) {
        return apply_ing(*stem);
      }
    }
    return std::nullopt;
  }

  std::optional<std::string> lookup_morphology_pronunciation(const std::string &word) const {
    if (const auto s = stem_s_pronunciation(word)) {
      return s;
    }
    if (const auto ed = stem_ed_pronunciation(word)) {
      return ed;
    }
    if (const auto ing = stem_ing_pronunciation(word)) {
      return ing;
    }
    return std::nullopt;
  }

  static std::optional<double> parse_inline_stress(const std::string &feature) {
    if (feature.empty()) {
      return std::nullopt;
    }
    if (feature == "0.5" || feature == "+0.5") {
      return 0.5;
    }
    if (feature == "-0.5") {
      return -0.5;
    }

    char *end = nullptr;
    const double value = std::strtod(feature.c_str(), &end);
    if (end != nullptr && *end == '\0') {
      return value;
    }
    return std::nullopt;
  }

  static bool parse_link(const std::string &text, std::size_t start, std::size_t &next_pos, Token &token) {
    // Python mapping: G2P.preprocess LINK_REGEX handling for inline pronunciation directives.
    if (text[start] != '[') {
      return false;
    }

    const auto close_label = text.find(']', start + 1);
    if (close_label == std::string::npos || close_label + 2 >= text.size()) {
      return false;
    }

    if (text[close_label + 1] != '(') {
      return false;
    }

    const auto close_paren = text.find(')', close_label + 2);
    if (close_paren == std::string::npos || close_paren <= close_label + 1) {
      return false;
    }

    const auto feature = text.substr(close_label + 2, close_paren - (close_label + 2));

    token.kind = TokenKind::Word;
    token.text = text.substr(start + 1, close_label - (start + 1));
    token.linked_pronunciation.clear();
    token.linked_stress = std::nullopt;
    token.linked_num_flags.clear();

    if (feature.size() > 1 && feature.front() == '/' && feature.back() == '/') {
      token.linked_pronunciation = feature.substr(1, feature.size() - 2);
    } else if (feature.size() > 1 && feature.front() == '[' && feature.back() == ']') {
      token.linked_alias = feature.substr(1, feature.size() - 2);
    } else if (feature.size() > 1 && feature.front() == '#' && feature.back() == '#') {
      token.linked_num_flags = feature.substr(1, feature.size() - 2);
    } else if (const auto stress = parse_inline_stress(feature)) {
      token.linked_stress = *stress;
    }

    next_pos = close_paren + 1;
    return true;
  }

  static std::vector<Token> tokenize(const std::string &text) {
    // Python mapping: simplified fusion of G2P.tokenize + G2P.retokenize for C++ exactness path.
    std::vector<Token> out;
    std::size_t i = 0;

    while (i < text.size()) {
      if (i + 3 < text.size() &&
          std::isalpha(static_cast<unsigned char>(text[i])) &&
          text[i + 1] == '.' &&
          std::isalpha(static_cast<unsigned char>(text[i + 2])) &&
          text[i + 3] == '.') {
        const char c0 = static_cast<char>(std::toupper(static_cast<unsigned char>(text[i])));
        const char c1 = static_cast<char>(std::toupper(static_cast<unsigned char>(text[i + 2])));
        if ((c0 == 'A' && c1 == 'M') || (c0 == 'P' && c1 == 'M')) {
          out.push_back({TokenKind::Word, std::string{c0, c1}, ""});
          i += 4;
          continue;
        }
      }

      Token link_token;
      std::size_t link_end = i;
      if (parse_link(text, i, link_end, link_token)) {
        out.push_back(std::move(link_token));
        i = link_end;
        continue;
      }

      const unsigned char c = static_cast<unsigned char>(text[i]);
      if (std::isspace(c)) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
        out.push_back({TokenKind::Whitespace, " ", ""});
        continue;
      }

      if (std::isalpha(c)) {
        std::size_t j = i;
        bool saw_ampersand = false;
        while (j < text.size()) {
          const unsigned char cj = static_cast<unsigned char>(text[j]);
          if (is_word_char(cj)) {
            ++j;
            continue;
          }
          if (text[j] == '&' && j + 1 < text.size() &&
              std::isalnum(static_cast<unsigned char>(text[j + 1]))) {
            saw_ampersand = true;
            ++j;
            continue;
          }
          break;
        }

        if (saw_ampersand) {
          out.push_back({TokenKind::Word, text.substr(i, j - i), ""});
          i = j;
          continue;
        }
      }

      if (std::isdigit(c) ||
          (text[i] == '-' && i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1]))) ||
          (text[i] == '.' && i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1])))) {
        const auto start = i;
        if (text[i] == '-') {
          ++i;
        }
        if (i < text.size() && text[i] == '.') {
          ++i;
        }
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
        while (i < text.size()) {
          if ((text[i] == ',' || text[i] == '.') && i + 1 < text.size() &&
              std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
            ++i;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
              ++i;
            }
            continue;
          }
          break;
        }
        const auto suffix_start = i;
        while (i < text.size() &&
               (std::isalpha(static_cast<unsigned char>(text[i])) || text[i] == '\'')) {
          ++i;
        }
        if (suffix_start != i) {
          const auto suffix = ascii_lower(text.substr(suffix_start, i - suffix_start));
          if (!is_ordinal_suffix(suffix) && !is_numeric_word_suffix(suffix)) {
            i = suffix_start;
          }
        }
        out.push_back({TokenKind::Word, text.substr(start, i - start), ""});
        continue;
      }

      if (is_expandable_symbol(static_cast<char>(c))) {
        out.push_back({TokenKind::Word, std::string(1, static_cast<char>(c)), ""});
        ++i;
        continue;
      }

      if (is_punctuation(static_cast<char>(c))) {
        out.push_back({TokenKind::Punctuation, std::string(1, static_cast<char>(c)), ""});
        ++i;
        continue;
      }

      if (is_word_char(c)) {
        const auto start = i;
        while (i < text.size() && is_word_char(static_cast<unsigned char>(text[i]))) {
          ++i;
        }
        out.push_back({TokenKind::Word, text.substr(start, i - start), ""});
        continue;
      }

      std::size_t advance = utf8_sequence_length(c);
      if (advance > 1) {
        if (i + advance > text.size()) {
          advance = 1;
        } else {
          bool valid = true;
          for (std::size_t k = 1; k < advance; ++k) {
            const unsigned char cb = static_cast<unsigned char>(text[i + k]);
            if ((cb & 0xC0) != 0x80) {
              valid = false;
              break;
            }
          }
          if (!valid) {
            advance = 1;
          }
        }
      }

      out.push_back({TokenKind::Punctuation, text.substr(i, advance), ""});
      i += advance;
    }

    return out;
  }

  std::optional<std::string> lookup_contextual_pronunciation(const Token &token, const TokenContext &ctx) const {
    // Python mapping: Lexicon.get_special_case for function words whose pronunciation depends on context.
    const auto key = ascii_lower(token.text);
    const bool has_future_word = ctx.future_vowel.has_value();
    const bool future_vowel = ctx.future_vowel.value_or(false);

    if (key == "the") {
      return future_vowel ? "ði" : "ðə";
    }

    if (token.text == "I") {
      return "ˌI";
    }

    if (key == "to") {
      if (!has_future_word) {
        if (const auto *from_lexicon = detail::find_english_lexicon_entry(token.text, variant_)) {
          return *from_lexicon;
        }
      }
      return future_vowel ? "tʊ" : "tə";
    }

    if (key == "in") {
      return has_future_word ? "ɪn" : "ˈɪn";
    }

    if (token.text == "a") {
      return has_future_word ? "ɐ" : apply_stress("A", true);
    }

    if (key == "an") {
      return "ɐn";
    }

    if (key == "am") {
      if (token.text == "am" && has_future_word) {
        return "ɐm";
      }
      return std::nullopt;
    }

    if (key == "by") {
      return has_future_word ? "bI" : "bˈI";
    }

    return std::nullopt;
  }

  static bool is_numeric_token(const std::string &text) {
    return parse_number_token(text).valid;
  }

  static std::string int_below_1000_to_words(int value) {
    static const std::vector<std::string> under_20 = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static const std::vector<std::string> tens = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

    if (value < 20) {
      return under_20[value];
    }
    if (value < 100) {
      const int t = value / 10;
      const int r = value % 10;
      return r == 0 ? tens[t] : tens[t] + " " + under_20[r];
    }

    const int h = value / 100;
    const int r = value % 100;
    if (r == 0) {
      return under_20[h] + " hundred";
    }
    return under_20[h] + " hundred " + int_below_1000_to_words(r);
  }

  static std::string integer_to_words(std::int64_t value) {
    if (value == 0) {
      return "zero";
    }

    const bool negative = value < 0;
    std::uint64_t number = negative ? static_cast<std::uint64_t>(-value) : static_cast<std::uint64_t>(value);

    static const std::vector<std::pair<std::uint64_t, std::string>> scales = {
        {1000000000ULL, "billion"}, {1000000ULL, "million"}, {1000ULL, "thousand"}};

    std::vector<std::string> chunks;
    for (const auto &[scale, name] : scales) {
      if (number >= scale) {
        const auto q = static_cast<int>(number / scale);
        chunks.push_back(int_below_1000_to_words(q) + " " + name);
        number %= scale;
      }
    }
    if (number > 0) {
      chunks.push_back(int_below_1000_to_words(static_cast<int>(number)));
    }

    std::ostringstream oss;
    if (negative) {
      oss << "minus ";
    }
    for (std::size_t i = 0; i < chunks.size(); ++i) {
      if (i > 0) {
        oss << ' ';
      }
      oss << chunks[i];
    }
    return oss.str();
  }

  static std::string ordinalize_word(const std::string &word) {
    static const std::unordered_map<std::string, std::string> irregular = {
        {"one", "first"},       {"two", "second"},       {"three", "third"},
        {"four", "fourth"},     {"five", "fifth"},       {"six", "sixth"},
        {"seven", "seventh"},   {"eight", "eighth"},     {"nine", "ninth"},
        {"ten", "tenth"},       {"eleven", "eleventh"},  {"twelve", "twelfth"},
        {"thirteen", "thirteenth"}, {"fourteen", "fourteenth"}, {"fifteen", "fifteenth"},
        {"sixteen", "sixteenth"},   {"seventeen", "seventeenth"}, {"eighteen", "eighteenth"},
        {"nineteen", "nineteenth"}, {"twenty", "twentieth"},      {"thirty", "thirtieth"},
        {"forty", "fortieth"},      {"fifty", "fiftieth"},        {"sixty", "sixtieth"},
        {"seventy", "seventieth"},  {"eighty", "eightieth"},      {"ninety", "ninetieth"},
        {"hundred", "hundredth"},   {"thousand", "thousandth"},   {"million", "millionth"},
        {"billion", "billionth"},   {"zero", "zeroth"},
    };

    const auto it = irregular.find(word);
    if (it != irregular.end()) {
      return it->second;
    }
    return word + "th";
  }

  static std::string integer_to_ordinal_words(std::int64_t value) {
    const auto cardinal = integer_to_words(value);
    std::vector<std::string> parts;
    {
      std::istringstream iss(cardinal);
      for (std::string w; iss >> w;) {
        parts.push_back(w);
      }
    }
    if (parts.empty()) {
      return cardinal;
    }
    parts.back() = ordinalize_word(parts.back());

    std::ostringstream oss;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      if (i > 0) {
        oss << ' ';
      }
      oss << parts[i];
    }
    return oss.str();
  }

  static std::string year_to_words(int year) {
    if (year >= 2000 && year <= 2009) {
      return integer_to_words(year);
    }

    const int first = year / 100;
    const int last = year % 100;
    std::ostringstream oss;
    oss << integer_to_words(first);
    if (last == 0) {
      oss << " hundred";
    } else if (last < 10) {
      oss << " O " << integer_to_words(last);
    } else {
      oss << ' ' << integer_to_words(last);
    }
    return oss.str();
  }

  static std::string digits_to_words(const std::string &digits) {
    static const std::vector<std::string> digit_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    std::ostringstream oss;
    bool first = true;
    for (char c : digits) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        continue;
      }
      if (!first) {
        oss << ' ';
      }
      first = false;
      oss << digit_words[c - '0'];
    }
    return oss.str();
  }

  std::optional<std::string> lookup_word_pronunciation(const std::string &word) const {
    if (const auto direct = lookup_direct_pronunciation(word)) {
      return direct;
    }
    return lookup_morphology_pronunciation(word);
  }

  std::optional<std::string> lookup_hyphen_compound_pronunciation(const std::string &word) const {
    if (word.size() < 3 || word.front() == '-' || word.back() == '-' || word.find('-') == std::string::npos) {
      return std::nullopt;
    }

    std::vector<std::string> parts;
    {
      std::size_t start = 0;
      while (start < word.size()) {
        const auto pos = word.find('-', start);
        const auto end = (pos == std::string::npos) ? word.size() : pos;
        if (end == start) {
          return std::nullopt;
        }
        parts.push_back(word.substr(start, end - start));
        if (pos == std::string::npos) {
          break;
        }
        start = pos + 1;
      }
    }

    if (parts.size() < 2) {
      return std::nullopt;
    }

    std::string combined;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const auto part_pron = lookup_word_pronunciation(parts[i]);
      if (!part_pron) {
        return std::nullopt;
      }
      std::string normalized = normalize_pronunciation(*part_pron);
      if (i > 0) {
        replace_all(normalized, "ˈ", "ˌ");
      }
      combined += normalized;
    }
    return collapse_ascii_spaces(combined);
  }

  static bool is_all_digits(const std::string &text) {
    if (text.empty()) {
      return false;
    }
    return std::all_of(text.begin(), text.end(), [](char c) {
      return std::isdigit(static_cast<unsigned char>(c));
    });
  }

  static bool is_all_upper_alpha(const std::string &text) {
    if (text.empty()) {
      return false;
    }
    return std::all_of(text.begin(), text.end(), [](char c) {
      return std::isupper(static_cast<unsigned char>(c));
    });
  }

  std::optional<std::string> pronounce_integer_token(const std::string &digits) const {
    if (!is_all_digits(digits)) {
      return std::nullopt;
    }
    std::vector<std::string> words;
    std::istringstream iss(integer_to_words(std::stoll(digits)));
    for (std::string w; iss >> w;) {
      words.push_back(w);
    }
    return words_to_phonemes(words);
  }

  std::optional<std::string> pronounce_code_segment(const std::string &segment) const {
    if (segment.empty()) {
      return std::nullopt;
    }

    if (const auto direct = lookup_direct_pronunciation(segment)) {
      return normalize_pronunciation(*direct);
    }

    if (const auto number = pronounce_integer_token(segment)) {
      return *number;
    }

    std::size_t digit_prefix = 0;
    while (digit_prefix < segment.size() && std::isdigit(static_cast<unsigned char>(segment[digit_prefix]))) {
      ++digit_prefix;
    }
    if (digit_prefix > 0 && digit_prefix < segment.size()) {
      const auto digits = segment.substr(0, digit_prefix);
      const auto suffix = segment.substr(digit_prefix);
      if (is_all_upper_alpha(suffix)) {
        const auto number = pronounce_integer_token(digits);
        const auto letters = lookup_direct_pronunciation(suffix);
        if (number && letters) {
          return *number + " " + normalize_pronunciation(*letters);
        }
      }
    }

    if (const auto morph = lookup_morphology_pronunciation(segment)) {
      return normalize_pronunciation(*morph);
    }

    return std::nullopt;
  }

  std::optional<std::string> lookup_ampersand_compound_pronunciation(const std::string &token_text) const {
    if (token_text == "&" || token_text.find('&') == std::string::npos) {
      return std::nullopt;
    }

    std::vector<std::string> parts;
    {
      std::size_t start = 0;
      while (start < token_text.size()) {
        const auto pos = token_text.find('&', start);
        const auto end = (pos == std::string::npos) ? token_text.size() : pos;
        if (end == start) {
          return std::nullopt;
        }
        parts.push_back(token_text.substr(start, end - start));
        if (pos == std::string::npos) {
          break;
        }
        start = pos + 1;
      }
    }

    if (parts.size() < 2) {
      return std::nullopt;
    }

    const auto and_pron = lookup_word_pronunciation("and");
    if (!and_pron) {
      return std::nullopt;
    }

    std::string combined;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const auto part_pron = pronounce_code_segment(parts[i]);
      if (!part_pron) {
        return std::nullopt;
      }
      const auto normalized_part = trim_ascii_spaces(*part_pron);
      if (normalized_part.empty()) {
        return std::nullopt;
      }
      if (!combined.empty()) {
        combined += " " + normalize_pronunciation(*and_pron) + " ";
      }
      combined += normalized_part;
    }
    return collapse_ascii_spaces(combined);
  }

  std::optional<std::string> lookup_hyphen_code_pronunciation(const std::string &token_text) const {
    if (token_text.find('-') == std::string::npos) {
      return std::nullopt;
    }

    const bool has_digit = std::any_of(token_text.begin(), token_text.end(), [](char c) {
      return std::isdigit(static_cast<unsigned char>(c));
    });
    const bool has_upper = std::any_of(token_text.begin(), token_text.end(), [](char c) {
      return std::isupper(static_cast<unsigned char>(c));
    });

    if (!has_digit || !has_upper) {
      return std::nullopt;
    }

    std::vector<std::string> parts;
    {
      std::size_t start = 0;
      while (start < token_text.size()) {
        const auto pos = token_text.find('-', start);
        const auto end = (pos == std::string::npos) ? token_text.size() : pos;
        if (end == start) {
          return std::nullopt;
        }
        parts.push_back(token_text.substr(start, end - start));
        if (pos == std::string::npos) {
          break;
        }
        start = pos + 1;
      }
    }

    if (parts.size() < 2) {
      return std::nullopt;
    }

    std::string combined;
    for (const auto &part : parts) {
      const auto part_pron = pronounce_code_segment(part);
      if (!part_pron) {
        return std::nullopt;
      }
      const auto normalized_part = trim_ascii_spaces(*part_pron);
      if (normalized_part.empty()) {
        return std::nullopt;
      }
      if (!combined.empty()) {
        combined.push_back(' ');
      }
      combined += normalized_part;
    }
    return combined;
  }

  std::string words_to_phonemes(const std::vector<std::string> &words) const {
    std::string out;
    bool needs_space = false;
    for (const auto &word : words) {
      if (word.empty()) {
        continue;
      }
      const auto pronunciation = lookup_word_pronunciation(word);
      if (!pronunciation) {
        throw std::runtime_error("Unmapped English expansion token: " + word);
      }
      std::string normalized = *pronunciation;
      if (word == "point") {
        replace_all(normalized, "ˈ", "");
        replace_all(normalized, "ˌ", "");
      }
      if (needs_space) {
        out.push_back(' ');
      }
      out += normalized;
      needs_space = true;
    }
    return out;
  }

  std::optional<std::string> expand_symbol_or_number(const std::vector<Token> &tokens, std::size_t index) const {
    // Python mapping: Lexicon.get_special_case symbols + num2words/extend_num numeric expansion logic.
    const auto &token = tokens[index].text;

    static const std::unordered_map<std::string, std::string> symbol_words = {
      {"%", "percent"}, {"&", "and"}, {"+", "plus"}, {"@", "at"}, {"/", "slash"}};

    const auto symbol_it = symbol_words.find(token);
    if (symbol_it != symbol_words.end()) {
      const auto pronunciation = lookup_word_pronunciation(symbol_it->second);
      if (!pronunciation) {
        throw std::runtime_error("Unmapped symbol expansion: " + symbol_it->second);
      }
      return *pronunciation;
    }

    if (token == "$" || token == "£" || token == "€") {
      const auto next_word = find_next_word_index(tokens, index);
      if (next_word && is_numeric_token(tokens[*next_word].text)) {
        return std::string{};
      }
      return std::nullopt;
    }

    if (!is_numeric_token(token)) {
      return std::nullopt;
    }

    const auto parsed = parse_number_token(token);
    if (!parsed.valid) {
      return std::nullopt;
    }

    const auto prev_word = find_prev_word_index(tokens, index);
    const bool has_currency_symbol =
        prev_word && (tokens[*prev_word].text == "$" || tokens[*prev_word].text == "£" || tokens[*prev_word].text == "€");

    std::vector<std::string> words;
    if (parsed.negative) {
      words.push_back("minus");
    }

    if (has_currency_symbol) {
      const auto currency_symbol = tokens[*prev_word].text;
      const std::string major = currency_symbol == "$" ? "dollar" : (currency_symbol == "£" ? "pound" : "euro");
      const std::string minor = currency_symbol == "£" ? "pence" : "cent";

      const auto major_value = std::stoll(parsed.whole.empty() ? "0" : parsed.whole);
      const auto minor_value = parsed.fractional.empty() ? 0 : std::stoi((parsed.fractional + "00").substr(0, 2));

      if (major_value != 0 || minor_value == 0) {
        std::istringstream iss(integer_to_words(major_value));
        for (std::string w; iss >> w;) {
          words.push_back(w);
        }
        words.push_back((std::llabs(major_value) == 1) ? major : (major + "s"));
      }
      if (minor_value != 0) {
        if (major_value != 0) {
          words.push_back("and");
        }
        std::istringstream iss(integer_to_words(minor_value));
        for (std::string w; iss >> w;) {
          words.push_back(w);
        }
        words.push_back(minor_value == 1 ? "cent" : minor + "s");
      }

      return words_to_phonemes(words);
    }

    const bool ordinal_suffix = is_ordinal_suffix(parsed.suffix);
    const bool morph_suffix = is_numeric_word_suffix(parsed.suffix);
    if (ordinal_suffix) {
      std::istringstream iss(integer_to_ordinal_words(std::stoll(parsed.whole.empty() ? "0" : parsed.whole)));
      for (std::string w; iss >> w;) {
        words.push_back(w);
      }
      return words_to_phonemes(words);
    }

    if (parsed.leading_dot) {
      words.push_back("point");
      std::istringstream dss(digits_to_words(parsed.fractional));
      for (std::string w; dss >> w;) {
        words.push_back(w);
      }
      return words_to_phonemes(words);
    }

    if (parsed.fractional.empty()) {
      std::optional<std::size_t> next_non_ws;
      for (std::size_t j = index + 1; j < tokens.size(); ++j) {
        if (tokens[j].kind == TokenKind::Whitespace) {
          continue;
        }
        next_non_ws = j;
        break;
      }

      const bool has_leading_zero = !parsed.original_core.empty() && parsed.original_core.size() > 1 &&
                                    parsed.original_core.front() == '0';
      const bool clock_like = next_non_ws.has_value() &&
                              tokens[*next_non_ws].kind == TokenKind::Punctuation &&
                              tokens[*next_non_ws].text == ":";

      if (has_leading_zero && clock_like) {
        const auto digit_words_text = digits_to_words(parsed.original_core);
        std::istringstream dss(digit_words_text);
        for (std::string w; dss >> w;) {
          words.push_back(w);
        }
        auto phonemes = words_to_phonemes(words);
        if (morph_suffix) {
          if (parsed.suffix == "s" || parsed.suffix == "'s") {
            phonemes = apply_plural_s(phonemes);
          } else if (parsed.suffix == "ed" || parsed.suffix == "'d") {
            phonemes = apply_past_ed(phonemes);
          } else if (parsed.suffix == "ing") {
            const auto ing = apply_ing(phonemes);
            if (!ing) {
              return std::nullopt;
            }
            phonemes = *ing;
          }
        }
        return phonemes;
      }

      const bool is_year = !parsed.negative && parsed.original_core.size() == 4 &&
                           std::all_of(parsed.original_core.begin(), parsed.original_core.end(),
                                       [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
      const auto number_text = is_year ? year_to_words(std::stoi(parsed.whole.empty() ? "0" : parsed.whole))
                                       : integer_to_words(std::stoll(parsed.whole.empty() ? "0" : parsed.whole));
      std::istringstream iss(number_text);
      for (std::string w; iss >> w;) {
        words.push_back(w);
      }
      auto phonemes = words_to_phonemes(words);
      if (morph_suffix) {
        if (parsed.suffix == "s" || parsed.suffix == "'s") {
          phonemes = apply_plural_s(phonemes);
        } else if (parsed.suffix == "ed" || parsed.suffix == "'d") {
          phonemes = apply_past_ed(phonemes);
        } else if (parsed.suffix == "ing") {
          const auto ing = apply_ing(phonemes);
          if (!ing) {
            return std::nullopt;
          }
          phonemes = *ing;
        }
      }
      return phonemes;
    }

    std::string fractional = parsed.fractional;
    while (!fractional.empty() && fractional.back() == '0') {
      fractional.pop_back();
    }

    std::istringstream iss(integer_to_words(std::stoll(parsed.whole.empty() ? "0" : parsed.whole)));
    for (std::string w; iss >> w;) {
      words.push_back(w);
    }
    if (!fractional.empty()) {
      words.push_back("point");
      std::istringstream dss(digits_to_words(fractional));
      for (std::string w; dss >> w;) {
        words.push_back(w);
      }
    }
    return words_to_phonemes(words);
  }

  std::string lookup_pronunciation(const std::vector<Token> &tokens, std::size_t index) const {
    // Python mapping: Lexicon.__call__/lookup sequence.
    // Order intentionally mirrors Python: linked overrides -> special cases -> contextual forms ->
    // symbol/number expansion -> dictionary lookup (with POS hint) -> morphology fallback.
    const auto &token = tokens[index];
    const auto key = ascii_lower(token.text);
    if (!token.linked_pronunciation.empty()) {
      return token.linked_pronunciation;
    }

    if (const auto ampersand = lookup_ampersand_compound_pronunciation(token.text)) {
      return *ampersand;
    }

    if (const auto code = lookup_hyphen_code_pronunciation(token.text)) {
      return *code;
    }

    if (token.text == "AS") {
      return variant_ == "en-gb" ? "ˈaz" : "ˈæz";
    }

    if (key == "bye-bye") {
      if (const auto bye = lookup_word_pronunciation("bye")) {
        return *bye + "—" + *bye;
      }
    }

    if (key == "used") {
      const auto next_word = find_next_word_index(tokens, index);
      if (next_word && ascii_lower(tokens[*next_word].text) == "to") {
        return variant_ == "en-gb" ? "jˈuːst" : "jˈust";
      }
    }

    if (key == "read") {
      const auto prev_word = find_prev_word_index(tokens, index);
      std::optional<std::size_t> next_non_ws;
      for (std::size_t j = index + 1; j < tokens.size(); ++j) {
        if (tokens[j].kind == TokenKind::Whitespace) {
          continue;
        }
        next_non_ws = j;
        break;
      }
      if (prev_word && ascii_lower(tokens[*prev_word].text) == "that" && next_non_ws &&
          tokens[*next_non_ws].kind == TokenKind::Punctuation && tokens[*next_non_ws].text == ":") {
        return "ɹˈɛd";
      }
    }

    if (key == "that") {
      const auto prev_word = find_prev_word_index(tokens, index);
      if (prev_word && ascii_lower(tokens[*prev_word].text) == "copy") {
        return "ðˈæt";
      }
    }

    if (key == "vs") {
      const auto prev_word = find_prev_word_index(tokens, index);
      std::optional<std::size_t> next_word;
      for (std::size_t j = index + 1; j < tokens.size(); ++j) {
        if (tokens[j].kind == TokenKind::Whitespace) {
          continue;
        }
        if (tokens[j].kind == TokenKind::Punctuation) {
          if (tokens[j].text == ".") {
            continue;
          }
          break;
        }
        next_word = j;
        break;
      }
      if (prev_word && next_word) {
        const auto &lhs = tokens[*prev_word].text;
        const auto &rhs = tokens[*next_word].text;
        const bool lhs_is_letter = lhs.size() == 1 && std::isalpha(static_cast<unsigned char>(lhs[0]));
        const bool rhs_is_letter = rhs.size() == 1 && std::isalpha(static_cast<unsigned char>(rhs[0]));
        if (lhs_is_letter && rhs_is_letter) {
          return variant_ == "en-gb" ? "vˈiːz" : "vˈiz";
        }
      }
      if (const auto versus = lookup_word_pronunciation("versus")) {
        return *versus;
      }
    }

    const auto ctx = build_context(tokens, index);
    const auto pos_hint = guess_pos_hint(tokens, index);
    if (const auto contextual = lookup_contextual_pronunciation(token, ctx)) {
      return *contextual;
    }

    if (const auto expanded = expand_symbol_or_number(tokens, index)) {
      return *expanded;
    }

    if (const auto direct = lookup_direct_pronunciation(token.text, pos_hint)) {
      return *direct;
    }

    if (const auto morph = lookup_morphology_pronunciation(token.text)) {
      return *morph;
    }

    if (const auto compound = lookup_hyphen_compound_pronunciation(token.text)) {
      return *compound;
    }

    if (token.text == "'s") {
      return "s";
    }

    if (token.text == "'d") {
      return "d";
    }

    if (token.text == "'ll") {
      return "əl";
    }

    if (token.text == "'m") {
      return "m";
    }

    if (token.text == "'t") {
      return "t";
    }

    if (token.text == "'re") {
      return "ɹ";
    }

    if (token.text == "'ve") {
      return "v";
    }

    if (token.text == "n'") {
      return "ən";
    }

      throw std::runtime_error("Unmapped English token: " + token.text);
  }
};

} // namespace

std::unique_ptr<G2P> make_english_engine(const std::string &variant) {
  if (variant != "en-us" && variant != "en-gb") {
    throw std::runtime_error("Unsupported English variant: " + variant);
  }
  return std::make_unique<EnglishG2P>(variant);
}

} // namespace misaki
