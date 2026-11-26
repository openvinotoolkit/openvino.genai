// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "model_identity.hpp"

namespace ov::genai {

namespace identity_memo_detail {

/// Pages of the weights file that are read to check a memo, spread evenly over the file.
constexpr size_t SAMPLE_PAGES = 64;
constexpr size_t SAMPLE_PAGE_BYTES = 64ull * 1024;

inline bool read_range(std::ifstream& file, uint64_t offset, size_t size, std::vector<char>& buffer) {
    buffer.resize(size);
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return static_cast<size_t>(file.gcount()) == size;
}

/// Digest of the whole file when it is small, otherwise of SAMPLE_PAGES pages of SAMPLE_PAGE_BYTES spread evenly over it.
inline std::optional<ContentHash> sampled_digest(const std::filesystem::path& path, uint64_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    std::vector<char> buffer;
    if (size <= SAMPLE_PAGES * SAMPLE_PAGE_BYTES) {
        if (!read_range(file, 0, static_cast<size_t>(size), buffer)) {
            return std::nullopt;
        }
        return compute_content_hash(buffer.data(), buffer.size());
    }
    std::vector<ContentHash> pages;
    for (size_t page = 0; page < SAMPLE_PAGES; ++page) {
        const uint64_t offset = (size - SAMPLE_PAGE_BYTES) / (SAMPLE_PAGES - 1) * page;
        if (!read_range(file, offset, SAMPLE_PAGE_BYTES, buffer)) {
            return std::nullopt;
        }
        pages.push_back(compute_content_hash(buffer.data(), buffer.size()));
    }
    return compute_content_hash(pages.data(), pages.size() * sizeof(ContentHash), size);
}

/// What tells that the weights file of a model is still the one a memo was written for: the sizes and modification
/// times of the graph and weights files, the whole graph file and sampled pages of the weights file.
inline std::optional<std::string> files_stamp(const std::filesystem::path& xml, const std::filesystem::path& bin) {
    std::error_code error;
    const uint64_t xml_size = std::filesystem::file_size(xml, error);
    if (error) {
        return std::nullopt;
    }
    const auto xml_time = std::filesystem::last_write_time(xml, error).time_since_epoch().count();
    if (error) {
        return std::nullopt;
    }
    const uint64_t bin_size = std::filesystem::file_size(bin, error);
    if (error) {
        return std::nullopt;
    }
    const auto bin_time = std::filesystem::last_write_time(bin, error).time_since_epoch().count();
    if (error) {
        return std::nullopt;
    }
    // The graph file is small next to the weights, so all of it is read.
    std::ifstream xml_file(xml, std::ios::binary);
    std::vector<char> xml_bytes;
    if (!xml_file || !read_range(xml_file, 0, static_cast<size_t>(xml_size), xml_bytes)) {
        return std::nullopt;
    }
    const std::optional<ContentHash> bin_sample = sampled_digest(bin, bin_size);
    if (!bin_sample.has_value()) {
        return std::nullopt;
    }
    const ContentHash xml_digest = compute_content_hash(xml_bytes.data(), xml_bytes.size());
    const std::string text = std::to_string(xml_size) + '|' + std::to_string(xml_time) + '|' + std::to_string(bin_size) + '|' +
                             std::to_string(bin_time) + '|' + std::to_string(model_identity_detail::IDENTITY_FORMAT_VERSION);
    const ContentHash parts[] = {compute_content_hash(text.data(), text.size()), xml_digest, *bin_sample};
    return model_identity_detail::to_hex(compute_content_hash(parts, sizeof(parts)));
}

constexpr const char* MEMO_HEADER = "ov_genai_weights_digest_memo_v1";

}  // namespace identity_memo_detail

/**
 * @brief Where the weights digest of the model whose graph file is @p xml is memoized under @p memo_directory.
 */
inline std::filesystem::path weights_digest_memo_path(const std::filesystem::path& memo_directory, const std::filesystem::path& xml) {
    std::error_code error;
    const std::string canonical = std::filesystem::weakly_canonical(xml, error).string();
    return memo_directory / model_identity_detail::to_hex(compute_content_hash(canonical.data(), canonical.size()));
}

/**
 * @brief compute_weights_digest() of a model read from @p model_directory, remembered in @p memo_directory.
 *
 * Hashing every weight takes seconds for a large model, so the digest is stored next to the stamp of the files it was
 * computed from. A later run that finds the same stamp (same sizes and modification times, same graph file, same
 * sampled pages of the weights file; about 4 MB read) takes the stored digest instead. Any mismatch, a missing or a
 * damaged memo computes the digest again and replaces the memo. A model that does not come from an IR file pair in
 * @p model_directory is not memoized.
 *
 * @param memo_hit Set to whether the stored digest was used.
 */
inline std::string compute_weights_digest_memoized(const std::shared_ptr<const ov::Model>& model,
                                                   const std::filesystem::path& model_directory,
                                                   const std::filesystem::path& memo_directory,
                                                   bool& memo_hit) {
    memo_hit = false;
    std::filesystem::path xml;
    for (const char* name : {"openvino_model.xml", "openvino_language_model.xml"}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(model_directory / name, error)) {
            xml = model_directory / name;
            break;
        }
    }
    if (xml.empty()) {
        return compute_weights_digest(model);
    }
    std::filesystem::path bin = xml;
    bin.replace_extension(".bin");
    const std::optional<std::string> stamp = identity_memo_detail::files_stamp(xml, bin);
    if (!stamp.has_value()) {
        return compute_weights_digest(model);
    }

    const std::filesystem::path memo_path = weights_digest_memo_path(memo_directory, xml);
    {
        std::ifstream memo(memo_path);
        std::string header, stored_stamp, stored_digest;
        if (memo && std::getline(memo, header) && std::getline(memo, stored_stamp) && std::getline(memo, stored_digest) &&
            header == identity_memo_detail::MEMO_HEADER && stored_stamp == *stamp && stored_digest.size() == 32) {
            memo_hit = true;
            return stored_digest;
        }
    }

    const std::string digest = compute_weights_digest(model);
    std::error_code error;
    std::filesystem::create_directories(memo_directory, error);
    if (!error) {
        // Written to a name of its own and renamed, so a concurrent run never reads half a memo.
        std::random_device random_device;
        const std::filesystem::path temporary =
            memo_path.string() + ".tmp" + std::to_string(random_device()) + std::to_string(random_device());
        {
            std::ofstream memo(temporary, std::ios::trunc);
            memo << identity_memo_detail::MEMO_HEADER << '\n' << *stamp << '\n' << digest << '\n';
        }
        std::filesystem::rename(temporary, memo_path, error);
        if (error) {
            std::filesystem::remove(temporary, error);
        }
    }
    return digest;
}

}  // namespace ov::genai
