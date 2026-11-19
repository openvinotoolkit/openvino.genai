// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov::genai {

/// Build-specific string both the in-tree default provider (AbiFileStoreProvider) and
/// StoreRegistry's handshake check must agree on byte-for-byte (see
/// OvKVCacheStoreModuleInfo::sdk_build_fingerprint in kv_cache_store_provider_abi.h). A real
/// deployment would derive this from the actual SDK/ABI build (version + platform + toolchain);
/// kept as a single literal, defined once to avoid the two sides drifting apart.
inline const char* get_host_sdk_build_fingerprint() {
    return "ov-genai-kvcache-store-abi-dev";
}

}  // namespace ov::genai
