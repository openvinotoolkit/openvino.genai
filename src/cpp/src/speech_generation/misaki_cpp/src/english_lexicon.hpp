#pragma once

#include <optional>
#include <string>

namespace misaki::detail {

const std::string *find_english_lexicon_entry(
    const std::string &word,
    const std::string &variant,
    const std::optional<std::string> &pos_tag_hint = std::nullopt);

} // namespace misaki::detail
