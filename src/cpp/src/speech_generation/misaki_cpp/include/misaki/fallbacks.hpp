// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "misaki/g2p.hpp"

namespace misaki {

class EspeakFallback {
public:
  explicit EspeakFallback(bool british,
                          std::string version = {},
                          std::string library_path = {});

  std::optional<std::string> operator()(const MToken& token) const;
  G2P::FallbackHook as_hook() const;
  bool backend_available() const;
  std::optional<std::string> backend_error() const;

private:
  bool british_ = false;
  std::string version_;
  std::string library_path_;
};

class EspeakG2P {
public:
  explicit EspeakG2P(std::string language,
                     std::string version = {},
                     std::string library_path = {});

  std::optional<std::string> phonemize(const std::string& text) const;
  bool backend_available() const;
  std::optional<std::string> backend_error() const;

private:
  std::string language_;
  std::string version_;
  std::string library_path_;
};

/// Raw IPA phonemization without Misaki normalization (no diphthong tie-merging, no
/// symbol remapping) — for consumers needing espeak-ng's individual IPA codepoints
/// untouched (e.g. Piper TTS, which encodes each IPA codepoint as a separate token id).
std::optional<std::string> raw_espeak_ipa_phonemize(const std::string& text,
                                                     const std::string& voice_name,
                                                     std::string library_path = {});

} // namespace misaki
