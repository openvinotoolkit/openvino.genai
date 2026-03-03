#include "english_lexicon.hpp"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef MISAKI_SOURCE_DIR
#define MISAKI_SOURCE_DIR "."
#endif

namespace misaki::detail {

using json = nlohmann::json;

class EnglishLexicon;

namespace {

std::mutex g_lexicon_mutex;
std::optional<std::filesystem::path> g_data_root_override;
std::unique_ptr<EnglishLexicon> g_us_lex;
std::unique_ptr<EnglishLexicon> g_gb_lex;

bool has_required_lexicon_files(const std::filesystem::path &root) {
  return std::filesystem::exists(root / "us_gold.json") &&
         std::filesystem::exists(root / "us_silver.json") &&
         std::filesystem::exists(root / "gb_gold.json") &&
         std::filesystem::exists(root / "gb_silver.json");
}

std::optional<std::filesystem::path> current_module_dir() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&current_module_dir),
                          &module)) {
    return std::nullopt;
  }
  std::wstring buffer(MAX_PATH, L'\0');
  DWORD len = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (len == 0) {
    return std::nullopt;
  }
  buffer.resize(len);
  return std::filesystem::path(buffer).parent_path();
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&current_module_dir), &info) == 0 || info.dli_fname == nullptr) {
    return std::nullopt;
  }
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

} // namespace

class EnglishLexicon {
public:
  explicit EnglishLexicon(std::string variant) {
    const auto data_root = resolve_data_root();
    const auto prefix = variant == "en-gb" ? "gb" : "us";
    load_json_file(data_root / (prefix + std::string("_gold.json")), true);
    load_json_file(data_root / (prefix + std::string("_silver.json")), false);
    grow_dictionary();
  }

  const std::string *find(const std::string &word, const std::optional<std::string> &pos_tag_hint) const {
    const auto direct = entries_.find(word);
    if (direct != entries_.end()) {
      return select_pronunciation(direct->second, pos_tag_hint);
    }
    const auto lower = ascii_lower(word);
    const auto lowered = entries_.find(lower);
    if (lowered != entries_.end()) {
      return select_pronunciation(lowered->second, pos_tag_hint);
    }
    return nullptr;
  }

private:
  struct PronunciationEntry {
    std::string default_pronunciation;
    std::unordered_map<std::string, std::string> tags;
  };

  std::unordered_map<std::string, PronunciationEntry> entries_;

  static std::filesystem::path resolve_data_root() {
    std::vector<std::filesystem::path> attempted_paths;
    attempted_paths.reserve(2);

    if (g_data_root_override) {
      attempted_paths.push_back(*g_data_root_override);
      if (has_required_lexicon_files(*g_data_root_override)) {
        return *g_data_root_override;
      }
    }

    if (const char *env_data_root = std::getenv("MISAKI_DATA_DIR")) {
      const auto env_path = std::filesystem::path(env_data_root);
      attempted_paths.push_back(env_path);
      if (has_required_lexicon_files(env_path)) {
        return env_path;
      }
    }

    std::ostringstream oss;
    oss << "Could not locate English lexicon data files (us/gb gold/silver json files)."
           " Supported lookup roots are:"
           "\n - set_english_lexicon_data_root(...) override"
           "\n - MISAKI_DATA_DIR environment variable"
           "\nTried:";
    for (const auto &path : attempted_paths) {
      oss << "\n - " << path.string();
    }

    throw std::runtime_error(oss.str());
  }

  static std::string ascii_lower(const std::string &input) {
    std::string out = input;
    for (auto &c : out) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
  }

  static std::string ascii_capitalize(const std::string &input) {
    if (input.empty()) {
      return input;
    }
    std::string out = ascii_lower(input);
    out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
  }

  static PronunciationEntry extract_pronunciation_entry(const json &value) {
    PronunciationEntry out;
    if (value.is_string()) {
      out.default_pronunciation = value.get<std::string>();
      return out;
    }
    if (!value.is_object()) {
      throw std::runtime_error("Unexpected pronunciation payload type");
    }

    for (auto v = value.begin(); v != value.end(); ++v) {
      if (v.value().is_string()) {
        out.tags.emplace(v.key(), v.value().get<std::string>());
      }
    }

    const auto def = out.tags.find("DEFAULT");
    if (def != out.tags.end()) {
      out.default_pronunciation = def->second;
    } else if (!out.tags.empty()) {
      out.default_pronunciation = out.tags.begin()->second;
    } else {
      throw std::runtime_error("No string pronunciation found in lexicon entry");
    }
    return out;
  }

  static const std::string *select_from_candidates(
      const PronunciationEntry &entry,
      const std::vector<std::string> &candidates) {
    for (const auto &tag : candidates) {
      const auto it = entry.tags.find(tag);
      if (it != entry.tags.end()) {
        return &it->second;
      }
    }
    return nullptr;
  }

  static const std::string *select_pronunciation(
      const PronunciationEntry &entry,
      const std::optional<std::string> &pos_tag_hint) {
    if (pos_tag_hint) {
      if (const auto *exact = select_from_candidates(entry, {*pos_tag_hint})) {
        return exact;
      }

      if (*pos_tag_hint == "VERB") {
        if (const auto *p = select_from_candidates(entry, {"VERB", "VB", "VBD", "VBN", "VBP", "VBZ", "VBG"})) {
          return p;
        }
      } else if (*pos_tag_hint == "NOUN") {
        if (const auto *p = select_from_candidates(entry, {"NOUN", "NN", "NNS", "NNP", "NNPS"})) {
          return p;
        }
      } else if (*pos_tag_hint == "ADJ") {
        if (const auto *p = select_from_candidates(entry, {"ADJ", "JJ", "JJR", "JJS"})) {
          return p;
        }
      } else if (*pos_tag_hint == "ADV") {
        if (const auto *p = select_from_candidates(entry, {"ADV", "RB", "RBR", "RBS"})) {
          return p;
        }
      }
    }

    if (!entry.default_pronunciation.empty()) {
      return &entry.default_pronunciation;
    }
    if (!entry.tags.empty()) {
      return &entry.tags.begin()->second;
    }
    return nullptr;
  }

  void load_json_file(const std::filesystem::path &path, bool overwrite_existing) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
      throw std::runtime_error("Failed to open English lexicon: " + path.string());
    }

    json parsed;
    in >> parsed;
    if (!parsed.is_object()) {
      throw std::runtime_error("English lexicon JSON root must be an object: " + path.string());
    }

    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      const auto pronunciation = extract_pronunciation_entry(it.value());
      if (overwrite_existing) {
        entries_[it.key()] = pronunciation;
      } else {
        entries_.emplace(it.key(), pronunciation);
      }
    }
  }

  void grow_dictionary() {
    std::vector<std::pair<std::string, PronunciationEntry>> additions;
    additions.reserve(entries_.size() / 4);

    for (const auto &kv : entries_) {
      const auto &key = kv.first;
      if (key.size() < 2) {
        continue;
      }
      const auto lower = ascii_lower(key);
      const auto capitalized = ascii_capitalize(lower);

      if (key == lower) {
        if (capitalized != key && entries_.find(capitalized) == entries_.end()) {
          additions.emplace_back(capitalized, kv.second);
        }
      } else if (key == capitalized) {
        if (entries_.find(lower) == entries_.end()) {
          additions.emplace_back(lower, kv.second);
        }
      }
    }

    for (const auto &kv : additions) {
      entries_.emplace(kv.first, kv.second);
    }
  }
};

const std::string *find_english_lexicon_entry(
    const std::string &word,
    const std::string &variant,
    const std::optional<std::string> &pos_tag_hint) {
  std::lock_guard<std::mutex> lock(g_lexicon_mutex);
  if (!g_us_lex) {
    g_us_lex = std::make_unique<EnglishLexicon>("en-us");
  }
  if (!g_gb_lex) {
    g_gb_lex = std::make_unique<EnglishLexicon>("en-gb");
  }
  return (variant == "en-gb" ? *g_gb_lex : *g_us_lex).find(word, pos_tag_hint);
}

void set_english_lexicon_data_root_override(const std::filesystem::path &path) {
  std::lock_guard<std::mutex> lock(g_lexicon_mutex);
  g_data_root_override = path;
  g_us_lex.reset();
  g_gb_lex.reset();
}

void clear_english_lexicon_data_root_override() {
  std::lock_guard<std::mutex> lock(g_lexicon_mutex);
  g_data_root_override = std::nullopt;
  g_us_lex.reset();
  g_gb_lex.reset();
}

} // namespace misaki::detail

namespace misaki {

void set_english_lexicon_data_root(const std::string &path) {
  detail::set_english_lexicon_data_root_override(std::filesystem::path(path));
}

void clear_english_lexicon_data_root() {
  detail::clear_english_lexicon_data_root_override();
}

} // namespace misaki
