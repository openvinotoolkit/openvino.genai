#include "misaki/fallbacks.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace misaki {
namespace {

constexpr int kEspeakCharsUtf8 = 1;
constexpr int kEspeakPhonemesIpa = 0x02;
constexpr int kEspeakPhonemesTie = 0x80;
constexpr int kEspeakInitializeDontExit = 0x8000;
constexpr int kAudioOutputSynchronous = 2;
constexpr int kEspeakOk = 0;

using FnEspeakInitialize = int (*)(int, int, const char *, int);
using FnEspeakSetVoiceByName = int (*)(const char *);
using FnEspeakTextToPhonemes = const char *(*)(const void **, int, int);

#ifdef _WIN32
using SharedLibraryHandle = HMODULE;
#else
using SharedLibraryHandle = void *;
#endif

struct EspeakApi {
  SharedLibraryHandle handle = nullptr;
  FnEspeakInitialize initialize = nullptr;
  FnEspeakSetVoiceByName set_voice_by_name = nullptr;
  FnEspeakTextToPhonemes text_to_phonemes = nullptr;
};

struct EspeakLoadState {
  std::optional<EspeakApi> api;
  std::string error;
};

std::string as_utf8(const char* value) {
  return std::string(value);
}

#if defined(__cpp_char8_t)
std::string as_utf8(const char8_t* value) {
  return std::string(reinterpret_cast<const char*>(value));
}
#endif

std::string trim(std::string value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.erase(value.begin());
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  return value;
}

void replace_all(std::string &text, const std::string &from, const std::string &to) {
  if (from.empty()) {
    return;
  }
  std::size_t start = 0;
  while ((start = text.find(from, start)) != std::string::npos) {
    text.replace(start, from.size(), to);
    start += to.size();
  }
}

SharedLibraryHandle open_shared_library(const std::string &path) {
#ifdef _WIN32
  if (path.empty()) {
    return nullptr;
  }
  return LoadLibraryA(path.c_str());
#else
  if (path.empty()) {
    return nullptr;
  }
  return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

void close_shared_library(SharedLibraryHandle handle) {
  if (!handle) {
    return;
  }
#ifdef _WIN32
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif
}

void *load_symbol(SharedLibraryHandle handle, const char *symbol) {
#ifdef _WIN32
  return reinterpret_cast<void *>(GetProcAddress(handle, symbol));
#else
  return dlsym(handle, symbol);
#endif
}

EspeakLoadState try_load_espeak_api(const std::string &library_hint) {
  std::vector<std::string> candidates;
  std::string attempted;
  if (!library_hint.empty()) {
    candidates.push_back(library_hint);
  }

#ifdef _WIN32
  candidates.emplace_back("libespeak-ng.dll");
  candidates.emplace_back("espeak-ng.dll");
#elif __APPLE__
  candidates.emplace_back("libespeak-ng.dylib");
#else
  candidates.emplace_back("libespeak-ng.so");
  candidates.emplace_back("libespeak-ng.so.1");
#endif

  for (const auto &candidate : candidates) {
    if (!attempted.empty()) {
      attempted += ", ";
    }
    attempted += candidate;

    SharedLibraryHandle handle = open_shared_library(candidate);
    if (!handle) {
      continue;
    }

    EspeakApi api;
    api.handle = handle;
    api.initialize = reinterpret_cast<FnEspeakInitialize>(load_symbol(handle, "espeak_Initialize"));
    api.set_voice_by_name = reinterpret_cast<FnEspeakSetVoiceByName>(load_symbol(handle, "espeak_SetVoiceByName"));
    api.text_to_phonemes = reinterpret_cast<FnEspeakTextToPhonemes>(load_symbol(handle, "espeak_TextToPhonemes"));

    if (!api.initialize || !api.set_voice_by_name || !api.text_to_phonemes) {
      close_shared_library(handle);
      continue;
    }

    const int sample_rate =
        api.initialize(kAudioOutputSynchronous, 0, nullptr, kEspeakInitializeDontExit);
    if (sample_rate <= 0) {
      close_shared_library(handle);
      continue;
    }

    return EspeakLoadState{std::move(api), {}};
  }

  std::string error = "Unable to load espeak-ng runtime library";
  if (!attempted.empty()) {
    error += " (candidates: " + attempted + ")";
  }
  return EspeakLoadState{std::nullopt, std::move(error)};
}

EspeakLoadState &get_cached_load_state(const std::string &library_path) {
  static std::mutex load_cache_mutex;
  static std::unordered_map<std::string, EspeakLoadState> load_cache;

  std::scoped_lock lock(load_cache_mutex);
  auto it = load_cache.find(library_path);
  if (it == load_cache.end()) {
    it = load_cache.emplace(library_path, try_load_espeak_api(library_path)).first;
  }
  return it->second;
}

std::string reorder_syllabic_marker(const std::string &text) {
  const std::string syllabic = "\xCC\xA9"; // U+0329
  const std::string schwa = as_utf8(u8"ᵊ");

  auto last_utf8_codepoint_start = [](const std::string &value) -> std::size_t {
    std::size_t start = value.size();
    while (start > 0) {
      --start;
      const unsigned char c = static_cast<unsigned char>(value[start]);
      if ((c & 0xC0) != 0x80) {
        break;
      }
    }
    return start;
  };

  std::string out;
  out.reserve(text.size());

  std::size_t i = 0;
  while (i < text.size()) {
    if (i + syllabic.size() <= text.size() && text.compare(i, syllabic.size(), syllabic) == 0) {
      if (!out.empty()) {
        const std::size_t cp_start = last_utf8_codepoint_start(out);
        const std::string last_cp = out.substr(cp_start);
        const bool is_ascii_space =
            last_cp.size() == 1 && std::isspace(static_cast<unsigned char>(last_cp[0]));
        if (!is_ascii_space) {
          out.erase(cp_start);
          out += schwa;
          out += last_cp;
        }
      }
      i += syllabic.size();
      continue;
    }

    out.push_back(text[i]);
    ++i;
  }

  replace_all(out, syllabic, "");
  return out;
}

std::string strip_language_switch_flags(std::string text) {
  auto is_ascii_alpha = [](char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    return (uc >= 'a' && uc <= 'z') || (uc >= 'A' && uc <= 'Z');
  };

  auto is_switch_code = [&](const std::string& token) {
    // Match espeak language-switch markers like (en), (fr), (pt-br), (en-us).
    // Keep conservative constraints to avoid stripping arbitrary parenthesized text.
    if (token.size() < 2 || token.size() > 12) {
      return false;
    }

    std::size_t alpha_count = 0;
    for (char c : token) {
      if (is_ascii_alpha(c)) {
        ++alpha_count;
        continue;
      }
      if (c == '-') {
        continue;
      }
      return false;
    }
    return alpha_count >= 2;
  };

  std::string out;
  out.reserve(text.size());

  std::size_t i = 0;
  while (i < text.size()) {
    if (text[i] != '(') {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    const std::size_t close_pos = text.find(')', i + 1);
    if (close_pos == std::string::npos) {
      out.push_back(text[i]);
      ++i;
      continue;
    }

    const std::string token = text.substr(i + 1, close_pos - i - 1);
    if (is_switch_code(token)) {
      i = close_pos + 1;
      continue;
    }

    out.append(text, i, close_pos - i + 1);
    i = close_pos + 1;
  }

  return out;
}

std::string normalize_espeak_to_misaki(std::string ps, bool british, const std::string &version) {
  static const std::vector<std::pair<std::string, std::string>> kE2M = {
      {as_utf8(u8"ʔˌn̩"), as_utf8(u8"ʔn")}, {as_utf8(u8"ʔn̩"), as_utf8(u8"ʔn")},
      {as_utf8(u8"a^ɪ"), as_utf8(u8"I")},   {as_utf8(u8"a^ʊ"), as_utf8(u8"W")},
      {as_utf8(u8"d^ʒ"), as_utf8(u8"ʤ")},   {as_utf8(u8"e^ɪ"), as_utf8(u8"A")},
      {as_utf8(u8"e"), as_utf8(u8"A")},     {as_utf8(u8"t^ʃ"), as_utf8(u8"ʧ")},
      {as_utf8(u8"ɔ^ɪ"), as_utf8(u8"Y")},   {as_utf8(u8"ə^l"), as_utf8(u8"ᵊl")},
      {as_utf8(u8"ʲo"), as_utf8(u8"jo")},   {as_utf8(u8"ʲə"), as_utf8(u8"jə")},
      {as_utf8(u8"ʲ"), as_utf8(u8"")},      {as_utf8(u8"ɚ"), as_utf8(u8"əɹ")},
      {as_utf8(u8"r"), as_utf8(u8"ɹ")},     {as_utf8(u8"x"), as_utf8(u8"k")},
      {as_utf8(u8"ç"), as_utf8(u8"k")},     {as_utf8(u8"ɐ"), as_utf8(u8"ə")},
      {as_utf8(u8"ɬ"), as_utf8(u8"l")},     {as_utf8(u8"̃"), as_utf8(u8"")},
  };

  for (const auto &mapping : kE2M) {
    replace_all(ps, mapping.first, mapping.second);
  }

  ps = reorder_syllabic_marker(ps);

  if (british) {
    replace_all(ps, as_utf8(u8"e^ə"), as_utf8(u8"ɛː"));
    replace_all(ps, as_utf8(u8"iə"), as_utf8(u8"ɪə"));
    replace_all(ps, as_utf8(u8"ə^ʊ"), as_utf8(u8"Q"));
  } else {
    replace_all(ps, as_utf8(u8"o^ʊ"), as_utf8(u8"O"));
    replace_all(ps, as_utf8(u8"ɜːɹ"), as_utf8(u8"ɜɹ"));
    replace_all(ps, as_utf8(u8"ɜː"), as_utf8(u8"ɜɹ"));
    replace_all(ps, as_utf8(u8"ɪə"), as_utf8(u8"iə"));
    replace_all(ps, as_utf8(u8"ː"), as_utf8(u8""));
  }

  replace_all(ps, as_utf8(u8"o"), as_utf8(u8"ɔ"));

  if (version != "2.0") {
    replace_all(ps, as_utf8(u8"ɾ"), as_utf8(u8"T"));
    replace_all(ps, as_utf8(u8"ʔ"), as_utf8(u8"t"));
  }

  replace_all(ps, "^", "");
  return trim(ps);
}

std::optional<std::string> raw_espeak_phonemize(EspeakApi &api,
                                                const std::string &text,
                                                const std::string &voice_name) {
  if (api.set_voice_by_name(voice_name.c_str()) != kEspeakOk) {
    return std::nullopt;
  }

  const void *text_ptr = static_cast<const void *>(text.c_str());
  std::string raw;

  const int phoneme_mode = kEspeakPhonemesIpa | kEspeakPhonemesTie | (static_cast<int>('^') << 8);
  while (text_ptr != nullptr) {
    const char *chunk = api.text_to_phonemes(&text_ptr, kEspeakCharsUtf8, phoneme_mode);
    if (!chunk) {
      break;
    }
    raw += chunk;
  }

  raw = trim(raw);
  if (raw.empty()) {
    return std::nullopt;
  }

  return raw;
}

std::string normalize_espeak_generic_to_misaki(std::string ps, const std::string &version) {
  static const std::vector<std::pair<std::string, std::string>> kE2M = {
      {as_utf8(u8"a^ɪ"), as_utf8(u8"I")},  {as_utf8(u8"a^ʊ"), as_utf8(u8"W")},
      {as_utf8(u8"d^z"), as_utf8(u8"ʣ")},  {as_utf8(u8"d^ʒ"), as_utf8(u8"ʤ")},
      {as_utf8(u8"e^ɪ"), as_utf8(u8"A")},  {as_utf8(u8"o^ʊ"), as_utf8(u8"O")},
      {as_utf8(u8"ə^ʊ"), as_utf8(u8"Q")},  {as_utf8(u8"s^s"), as_utf8(u8"S")},
      {as_utf8(u8"t^s"), as_utf8(u8"ʦ")},  {as_utf8(u8"t^ʃ"), as_utf8(u8"ʧ")},
      {as_utf8(u8"ɔ^ɪ"), as_utf8(u8"Y")},
  };

  static const std::vector<std::pair<std::string, std::string>> kE2M_v20 = {
      {as_utf8(u8"œ̃"), as_utf8(u8"B")}, {as_utf8(u8"ɔ̃"), as_utf8(u8"C")},
      {as_utf8(u8"ɑ̃"), as_utf8(u8"D")}, {as_utf8(u8"ɛ̃"), as_utf8(u8"E")},
      {as_utf8(u8"ʊ̃"), as_utf8(u8"V")}, {as_utf8(u8"ũ"), as_utf8(u8"U")},
      {as_utf8(u8"õ"), as_utf8(u8"X")}, {as_utf8(u8"ɐ̃"), as_utf8(u8"Z")},
  };

  replace_all(ps, as_utf8(u8"«"), as_utf8(u8"“"));
  replace_all(ps, as_utf8(u8"»"), as_utf8(u8"”"));

  for (const auto &mapping : kE2M) {
    replace_all(ps, mapping.first, mapping.second);
  }
  if (version == "2.0") {
    for (const auto &mapping : kE2M_v20) {
      replace_all(ps, mapping.first, mapping.second);
    }
  }

  replace_all(ps, "^", "");

  if (version == "2.0") {
    replace_all(ps, as_utf8(u8"\u0329"), "");
    replace_all(ps, as_utf8(u8"\u032A"), "");
    ps = reorder_syllabic_marker(ps);
  } else {
    replace_all(ps, "-", "");
  }

  // Python parity: EspeakBackend(..., language_switch='remove-flags').
  ps = strip_language_switch_flags(ps);

  replace_all(ps, as_utf8(u8"«"), "(");
  replace_all(ps, as_utf8(u8"»"), ")");
  return trim(ps);
}

std::optional<std::string> phonemize_with_espeak_api(EspeakApi &api,
                                                     const std::string &text,
                                                     bool british,
                                                     const std::string &version) {
  std::vector<std::string> voice_candidates;
  if (british) {
    voice_candidates = {"en-gb", "en", "en-us"};
  } else {
    voice_candidates = {"en-us", "en", "en-gb"};
  }

  std::optional<std::string> raw;
  for (const auto& voice_name : voice_candidates) {
    raw = raw_espeak_phonemize(api, text, voice_name);
    if (raw.has_value()) {
      break;
    }
  }

  if (!raw.has_value()) {
    return std::nullopt;
  }

  const auto normalized = normalize_espeak_to_misaki(*raw, british, version);
  if (normalized.empty()) {
    return std::nullopt;
  }
  return normalized;
}

std::optional<std::string> phonemize_generic_with_espeak_api(EspeakApi &api,
                                                             std::string text,
                                                             const std::string &language,
                                                             const std::string &version) {
  auto lower_copy = [](std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return value;
  };

  auto build_voice_candidates = [&](const std::string& variant) {
    std::vector<std::string> candidates;
    const std::string normalized = lower_copy(variant);

    auto add = [&](const std::string& candidate) {
      if (candidate.empty()) {
        return;
      }
      if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
        candidates.push_back(candidate);
      }
    };

    add(normalized);

    if (normalized == "fr-fr") {
      add("fr");
      add("fr_fr");
    } else if (normalized == "pt-br") {
      add("pt");
      add("pt_br");
    } else if (normalized == "en-gb") {
      add("en");
      add("en-gb-x-gbclan");
    } else if (normalized == "en-us") {
      add("en");
      add("en-us");
    } else {
      const auto dash = normalized.find('-');
      if (dash != std::string::npos) {
        add(normalized.substr(0, dash));
      }
    }

    return candidates;
  };

  auto is_preserved_punctuation = [](char c) {
    switch (c) {
    case '.':
    case ',':
    case '!':
    case '?':
    case ':':
    case ';':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
      return true;
    default:
      return false;
    }
  };

  auto count_leading_spaces = [](const std::string& value) {
    std::size_t count = 0;
    while (count < value.size() && std::isspace(static_cast<unsigned char>(value[count]))) {
      ++count;
    }
    return count;
  };

  auto count_trailing_spaces = [](const std::string& value) {
    std::size_t count = 0;
    while (count < value.size() && std::isspace(static_cast<unsigned char>(value[value.size() - 1 - count]))) {
      ++count;
    }
    return count;
  };

  std::vector<std::string> chunks;
  chunks.reserve(text.size());

  std::string current;
  for (char c : text) {
    if (is_preserved_punctuation(c)) {
      if (!current.empty()) {
        chunks.push_back(current);
        current.clear();
      }
      chunks.push_back(std::string(1, c));
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    chunks.push_back(current);
  }

  std::string out;
  for (const auto& chunk : chunks) {
    if (chunk.size() == 1 && is_preserved_punctuation(chunk[0])) {
      out += chunk;
      continue;
    }

    const std::size_t leading_spaces = count_leading_spaces(chunk);
    const std::size_t trailing_spaces = count_trailing_spaces(chunk);
    const std::size_t core_end = chunk.size() - trailing_spaces;
    const std::string core =
        leading_spaces < core_end ? chunk.substr(leading_spaces, core_end - leading_spaces) : std::string{};

    if (core.empty()) {
      out += chunk;
      continue;
    }

    std::optional<std::string> raw;
    for (const auto& voice_candidate : build_voice_candidates(language)) {
      raw = raw_espeak_phonemize(api, core, voice_candidate);
      if (raw.has_value()) {
        break;
      }
    }

    if (!raw.has_value()) {
      return std::nullopt;
    }

    const auto normalized = normalize_espeak_generic_to_misaki(*raw, version);
    if (normalized.empty()) {
      return std::nullopt;
    }

    out.append(leading_spaces, ' ');
    out += normalized;
    out.append(trailing_spaces, ' ');
  }

  const auto normalized_out = trim(out);
  if (normalized_out.empty()) {
    return std::nullopt;
  }
  return normalized_out;
}

} // namespace

EspeakFallback::EspeakFallback(bool british, std::string version, std::string library_path)
    : british_(british), version_(std::move(version)), library_path_(std::move(library_path)) {
}

std::optional<std::string> EspeakFallback::operator()(const MToken &token) const {
  const std::string input = trim(token.text);
  if (input.empty()) {
    return std::nullopt;
  }

  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);

  auto &state = get_cached_load_state(library_path_);
  if (!state.api.has_value()) {
    return std::nullopt;
  }

  return phonemize_with_espeak_api(*(state.api), input, british_, version_);
}

G2P::FallbackHook EspeakFallback::as_hook() const {
  return [self = *this](const MToken &token) -> std::optional<std::string> {
    return self(token);
  };
}

bool EspeakFallback::backend_available() const {
  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);
  return get_cached_load_state(library_path_).api.has_value();
}

std::optional<std::string> EspeakFallback::backend_error() const {
  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);
  const auto &state = get_cached_load_state(library_path_);
  if (state.api.has_value() || state.error.empty()) {
    return std::nullopt;
  }
  return state.error;
}

EspeakG2P::EspeakG2P(std::string language, std::string version, std::string library_path)
    : language_(std::move(language)), version_(std::move(version)), library_path_(std::move(library_path)) {
}

std::optional<std::string> EspeakG2P::phonemize(const std::string &text) const {
  const std::string input = trim(text);
  if (input.empty()) {
    return std::nullopt;
  }

  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);

  auto &state = get_cached_load_state(library_path_);
  if (!state.api.has_value()) {
    return std::nullopt;
  }

  return phonemize_generic_with_espeak_api(*(state.api), input, language_, version_);
}

bool EspeakG2P::backend_available() const {
  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);
  return get_cached_load_state(library_path_).api.has_value();
}

std::optional<std::string> EspeakG2P::backend_error() const {
  static std::mutex api_mutex;
  std::scoped_lock lock(api_mutex);
  const auto &state = get_cached_load_state(library_path_);
  if (state.api.has_value() || state.error.empty()) {
    return std::nullopt;
  }
  return state.error;
}

} // namespace misaki
