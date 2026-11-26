// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "continuous_batching/cache/abi_file_store_provider.hpp"
#include "continuous_batching/cache/abi_kv_cache_store.hpp"
#include "openvino/core/except.hpp"

using namespace ov::genai;

namespace {

constexpr size_t BLOCK_SIZE = 64;

std::unique_ptr<AbiKVCacheStore> open_store(size_t num_blocks, bool pass_capabilities = true) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};

    OvKVCacheStoreOpenOptions options{};
    options.directory = nullptr;  // system temp directory
    options.capacity_bytes = BLOCK_SIZE * num_blocks;
    options.properties = properties;
    options.property_count = 1;

    OvKVCacheStoreOpenResult result{};
    const OvKVCacheStatus status = provider->vtable->open(provider->self, nullptr, &options, &result);
    EXPECT_EQ(status, OV_KVCACHE_OK);
    EXPECT_EQ(result.status, OV_KVCACHE_OK);
    EXPECT_NE(result.vtable, nullptr);
    EXPECT_NE(result.store, nullptr);

    AbiKVCacheStore::StoreInfo info;
    if (pass_capabilities) {
        info.capabilities = provider->capabilities;
        if (result.has_semantics_fingerprint != 0) {
            info.semantics_fingerprint = result.store_semantics_fingerprint;
        }
    }
    return std::make_unique<AbiKVCacheStore>(result.vtable, result.store, "kv", BLOCK_SIZE, num_blocks, info);
}

std::vector<uint8_t> make_block(uint8_t seed) {
    std::vector<uint8_t> data(BLOCK_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}


// The plugin is addressed by digest, so the tests that name a block by one number give it a digest made of that number.
PrefixKey key_of(size_t hash) {
    return PrefixKey{hash, ContentHash{hash, hash}};
}

bool has_one(const AbiKVCacheStore& store, size_t hash) {
    return store.has_blocks({key_of(hash)})[0];
}

bool write_one(AbiKVCacheStore& store, size_t hash, const std::vector<uint8_t>& data) {
    return store.write_blocks({key_of(hash)}, {&data})[0];
}

bool read_one(const AbiKVCacheStore& store, size_t hash, std::vector<uint8_t>& data) {
    std::vector<std::vector<uint8_t>> blocks;
    if (!store.read_blocks({key_of(hash)}, blocks)[0]) {
        return false;
    }
    data = std::move(blocks[0]);
    return true;
}

void drop_one(AbiKVCacheStore& store, size_t hash) {
    store.invalidate(key_of(hash));
}

std::vector<PrefixKey> keys_of(const std::vector<size_t>& hashes) {
    std::vector<PrefixKey> keys;
    for (const size_t hash : hashes) {
        keys.push_back(key_of(hash));
    }
    return keys;
}

std::vector<bool> has_many(const AbiKVCacheStore& store, const std::vector<size_t>& hashes) {
    return store.has_blocks(keys_of(hashes));
}

std::vector<bool> write_many(AbiKVCacheStore& store,
                             const std::vector<size_t>& hashes,
                             const std::vector<const std::vector<uint8_t>*>& data) {
    return store.write_blocks(keys_of(hashes), data);
}

std::vector<bool> read_many(const AbiKVCacheStore& store,
                            const std::vector<size_t>& hashes,
                            std::vector<std::vector<uint8_t>>& data) {
    return store.read_blocks(keys_of(hashes), data);
}

}  // namespace

TEST(TestAbiFileStoreProvider, ModuleHandshakeIsWellFormed) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    ASSERT_NE(module_info, nullptr);
    EXPECT_EQ(module_info->abi_major, static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MAJOR));
    EXPECT_LE(module_info->abi_minor, static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MINOR));
    ASSERT_EQ(module_info->provider_count, 1u);

    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);
    ASSERT_NE(provider, nullptr);
    EXPECT_STREQ(provider->name, "file");
    EXPECT_NE(provider->vtable, nullptr);
    EXPECT_EQ(provider->vtable->struct_bytes, sizeof(OvKVCacheStoreProviderVTable));
    EXPECT_EQ(module_info->get_provider(1), nullptr);
}

TEST(TestAbiFileStoreProvider, WriteThenReadRoundTripsBytesThroughTheAbi) {
    auto store = open_store(/*num_blocks=*/4);
    const auto expected = make_block(/*seed=*/7);

    EXPECT_FALSE(has_one(*store, 42));
    EXPECT_TRUE(write_one(*store, 42, expected));
    EXPECT_TRUE(has_one(*store, 42));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_one(*store, 42, actual));
    EXPECT_EQ(actual, expected);
}

TEST(TestAbiFileStoreProvider, HasBlocksAnswersForTheWholeChainInOrder) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    ASSERT_TRUE(write_one(*store, 3, make_block(3)));

    EXPECT_EQ(has_many(*store, {1, 2, 3, 4}), (std::vector<bool>{true, false, true, false}));
    EXPECT_TRUE(has_many(*store, {}).empty());
}

TEST(TestAbiFileStoreProvider, AdvertisesAsyncIoAndASemanticsFingerprint) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    EXPECT_NE(provider->capabilities & OV_KVCACHE_CAP_ASYNC_IO, 0u);
    EXPECT_NE(provider->capabilities & OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT, 0u);

    auto store = open_store(/*num_blocks=*/2);
    EXPECT_TRUE(store->uses_async_io());
    EXPECT_TRUE(store->get_info().semantics_fingerprint.has_value());
}

TEST(TestAbiFileStoreProvider, StaysOnSynchronousIoWhenTheCapabilityIsNotPassed) {
    auto store = open_store(/*num_blocks=*/2, /*pass_capabilities=*/false);
    EXPECT_FALSE(store->uses_async_io());

    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_one(*store, 1, actual));
    EXPECT_EQ(actual, make_block(1));
}

TEST(TestAbiFileStoreProvider, ReadBlocksReportsEveryBlockOfTheBatchInOrder) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    ASSERT_TRUE(write_one(*store, 3, make_block(3)));

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(read_many(*store, {1, 2, 3}, data), (std::vector<bool>{true, false, true}));
    ASSERT_EQ(data.size(), 3u);
    EXPECT_EQ(data[0], make_block(1));
    EXPECT_TRUE(data[1].empty());
    EXPECT_EQ(data[2], make_block(3));

    EXPECT_TRUE(read_many(*store, {}, data).empty());
    EXPECT_TRUE(data.empty());
}

namespace {

/// Wraps the file provider's store vtable to count the async batches and to make every first `wait` answer PENDING.
struct AsyncSpy {
    static inline const OvKVCacheStoreVTable* inner = nullptr;
    static inline size_t submitted = 0;
    static inline size_t terminal_waits = 0;
    static inline size_t sync_reads = 0;
    static inline size_t sync_writes = 0;
    static inline OvKVCacheAsyncToken pending_once = 0;

    static OvKVCacheStatus read_blocks(void* self, const OvKVCacheStoreReadRequest* r, OvKVCacheStoreReadResult* o, size_t n) {
        ++sync_reads;
        return inner->read_blocks(self, r, o, n);
    }
    static OvKVCacheStatus write_blocks(void* self, const OvKVCacheStoreWriteRequest* r, OvKVCacheStoreWriteResult* o, size_t n) {
        ++sync_writes;
        return inner->write_blocks(self, r, o, n);
    }
    static OvKVCacheStatus read_async(void* self,
                                      const OvKVCacheStoreReadRequest* r,
                                      OvKVCacheStoreReadResult* o,
                                      size_t n,
                                      OvKVCacheAsyncToken* token) {
        ++submitted;
        return inner->read_blocks_async(self, r, o, n, token);
    }
    static OvKVCacheStatus write_async(void* self,
                                       const OvKVCacheStoreWriteRequest* r,
                                       OvKVCacheStoreWriteResult* o,
                                       size_t n,
                                       OvKVCacheAsyncToken* token) {
        ++submitted;
        return inner->write_blocks_async(self, r, o, n, token);
    }
    static OvKVCacheStatus wait(void* self, OvKVCacheAsyncToken token, uint32_t timeout_ms) {
        if (pending_once != token) {
            pending_once = token;
            return OV_KVCACHE_PENDING;
        }
        pending_once = 0;
        ++terminal_waits;
        return inner->wait(self, token, timeout_ms);
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, EveryAsyncBatchIsWaitedOnUntilItIsTerminal) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    AsyncSpy::inner = opened.vtable;
    AsyncSpy::submitted = AsyncSpy::terminal_waits = AsyncSpy::sync_reads = AsyncSpy::sync_writes = 0;
    AsyncSpy::pending_once = 0;
    OvKVCacheStoreVTable spied = *opened.vtable;
    spied.read_blocks = &AsyncSpy::read_blocks;
    spied.write_blocks = &AsyncSpy::write_blocks;
    spied.read_blocks_async = &AsyncSpy::read_async;
    spied.write_blocks_async = &AsyncSpy::write_async;
    spied.wait = &AsyncSpy::wait;

    {
        AbiKVCacheStore store(&spied, opened.store, "kv", BLOCK_SIZE, 4, {provider->capabilities, std::nullopt});
        ASSERT_TRUE(store.uses_async_io());
        ASSERT_TRUE(write_one(store, 1, make_block(1)));
        ASSERT_TRUE(write_one(store, 2, make_block(2)));
        std::vector<std::vector<uint8_t>> data;
        EXPECT_EQ(read_many(store, {1, 2}, data), (std::vector<bool>{true, true}));
        EXPECT_EQ(data[1], make_block(2));
        std::vector<uint8_t> single;
        EXPECT_TRUE(read_one(store, 1, single));
    }

    EXPECT_EQ(AsyncSpy::submitted, 4u);
    EXPECT_EQ(AsyncSpy::terminal_waits, 4u) << "each submitted batch must be waited on until it reports a terminal status";
    EXPECT_EQ(AsyncSpy::sync_reads + AsyncSpy::sync_writes, 0u) << "the synchronous entries are not used when async is advertised";
}

TEST(TestAbiFileStoreProvider, ReadReportsMissForUnknownHash) {
    auto store = open_store(/*num_blocks=*/4);
    std::vector<uint8_t> data;
    EXPECT_FALSE(read_one(*store, 1234, data));
}

namespace {

/// Returns the stored blocks with a damaged witness, the way a corrupted store would.
struct CorruptWitnessPlugin {
    static inline const OvKVCacheStoreVTable* inner = nullptr;

    static OvKVCacheStatus read_blocks(void* self, const OvKVCacheStoreReadRequest* r, OvKVCacheStoreReadResult* o, size_t n) {
        const OvKVCacheStatus status = inner->read_blocks(self, r, o, n);
        for (size_t i = 0; i < n; ++i) {
            if (o[i].status == OV_KVCACHE_OK && o[i].witness.len > 0) {
                static_cast<uint8_t*>(r[i].witness_out.data)[0] ^= 0xFF;
            }
        }
        return status;
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, ABlockWhoseWitnessDoesNotMatchIsDroppedAndReadAsAMiss) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    CorruptWitnessPlugin::inner = opened.vtable;
    OvKVCacheStoreVTable corrupting = *opened.vtable;
    corrupting.read_blocks = &CorruptWitnessPlugin::read_blocks;

    AbiKVCacheStore store(&corrupting, opened.store, "kv", BLOCK_SIZE, 4, AbiKVCacheStore::StoreInfo{});
    ASSERT_FALSE(store.uses_async_io());
    ASSERT_TRUE(write_one(store, 1, make_block(1)));
    ASSERT_TRUE(has_one(store, 1));

    std::vector<std::vector<uint8_t>> data;
    EXPECT_NO_THROW(EXPECT_EQ(read_many(store, {1}, data), (std::vector<bool>{false})));
    EXPECT_TRUE(data[0].empty());
    EXPECT_FALSE(has_one(store, 1)) << "a block that failed its witness check is dropped from the store";
}

namespace {

/// A plugin whose async batches never finish on their own: `wait` reports PENDING until the host cancels the token.
struct StalledPlugin {
    static inline size_t waits = 0;
    static inline size_t cancels = 0;

    static OvKVCacheStatus wait(void* /*self*/, OvKVCacheAsyncToken /*token*/, uint32_t /*timeout_ms*/) {
        ++waits;
        return cancels == 0 ? OV_KVCACHE_PENDING : OV_KVCACHE_ERR_CANCELLED;
    }
    static OvKVCacheStatus cancel(void* /*self*/, OvKVCacheAsyncToken /*token*/) {
        ++cancels;
        return OV_KVCACHE_OK;
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, ABatchThatStallsIsCancelledAndFailsInsteadOfBlocking) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    StalledPlugin::waits = StalledPlugin::cancels = 0;
    OvKVCacheStoreVTable stalled = *opened.vtable;
    stalled.wait = &StalledPlugin::wait;
    stalled.cancel = &StalledPlugin::cancel;

    AbiKVCacheStore store(&stalled, opened.store, "kv", BLOCK_SIZE, 4, {provider->capabilities, std::nullopt},
                          /*io_timeout_ms=*/20);
    ASSERT_TRUE(store.uses_async_io());
    EXPECT_FALSE(write_one(store, 1, make_block(1))) << "a cancelled batch writes nothing and is not an error";
    EXPECT_EQ(StalledPlugin::cancels, 1u);
    EXPECT_EQ(StalledPlugin::waits, 2u) << "one wait that times out, then the wait for the cancelled token";

    StalledPlugin::waits = StalledPlugin::cancels = 0;
    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(read_many(store, {1}, data), (std::vector<bool>{false})) << "a cancelled batch reads nothing";
    EXPECT_EQ(StalledPlugin::cancels, 1u);
    EXPECT_FALSE(store.is_given_up()) << "a cancelled batch that ends in time leaves the store usable";
}

namespace {

/// A plugin that never ends an async batch, even after a cancel, and keeps the buffers of the batch it was given.
struct UnresponsivePlugin {
    static inline const OvKVCacheStoreVTable* inner = nullptr;
    static inline size_t submitted = 0;
    static inline size_t waits = 0;
    static inline size_t cancels = 0;
    static inline size_t probes = 0;
    static inline size_t flushes = 0;
    static inline size_t destroys = 0;
    static inline void* held_buffer = nullptr;
    static inline size_t held_bytes = 0;
    static inline const void* held_source = nullptr;

    static void reset() {
        submitted = waits = cancels = probes = flushes = destroys = 0;
        held_buffer = nullptr;
        held_bytes = 0;
        held_source = nullptr;
    }

    static OvKVCacheStatus read_async(void* /*self*/,
                                      const OvKVCacheStoreReadRequest* reqs,
                                      OvKVCacheStoreReadResult* /*out*/,
                                      size_t /*count*/,
                                      OvKVCacheAsyncToken* token) {
        ++submitted;
        held_buffer = reqs[0].target.base;
        held_bytes = reqs[0].target.bytes;
        *token = 1;
        return OV_KVCACHE_OK;
    }
    static OvKVCacheStatus write_async(void* /*self*/,
                                       const OvKVCacheStoreWriteRequest* reqs,
                                       OvKVCacheStoreWriteResult* /*out*/,
                                       size_t /*count*/,
                                       OvKVCacheAsyncToken* token) {
        ++submitted;
        held_source = reqs[0].source;
        *token = 1;
        return OV_KVCACHE_OK;
    }
    static OvKVCacheStatus wait(void* /*self*/, OvKVCacheAsyncToken /*token*/, uint32_t /*timeout_ms*/) {
        ++waits;
        return OV_KVCACHE_PENDING;
    }
    static OvKVCacheStatus cancel(void* /*self*/, OvKVCacheAsyncToken /*token*/) {
        ++cancels;
        return OV_KVCACHE_OK;
    }
    static OvKVCacheStatus has_blocks(void* self, const OvKVCacheBlockKey* keys, uint8_t* present, size_t count) {
        ++probes;
        return inner->has_blocks(self, keys, present, count);
    }
    static OvKVCacheStatus flush(void* self) {
        ++flushes;
        return inner->flush(self);
    }
    static void destroy(void* self) {
        ++destroys;
        inner->destroy(self);
    }
};

struct OpenedFileStore {
    const OvKVCacheStoreProviderDescriptor* provider = nullptr;
    OvKVCacheStoreOpenResult opened{};
};

OpenedFileStore open_file_store(size_t num_blocks) {
    OpenedFileStore result;
    result.provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * num_blocks;
    options.properties = properties;
    options.property_count = 1;
    EXPECT_EQ(result.provider->vtable->open(result.provider->self, nullptr, &options, &result.opened), OV_KVCACHE_OK);
    return result;
}

}  // namespace

TEST(TestAbiFileStoreProvider, APluginThatIgnoresCancelLeavesAMissAndAGivenUpStoreInsteadOfHanging) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    UnresponsivePlugin::reset();
    UnresponsivePlugin::inner = file_store.opened.vtable;
    OvKVCacheStoreVTable unresponsive = *file_store.opened.vtable;
    unresponsive.read_blocks_async = &UnresponsivePlugin::read_async;
    unresponsive.wait = &UnresponsivePlugin::wait;
    unresponsive.cancel = &UnresponsivePlugin::cancel;
    unresponsive.has_blocks = &UnresponsivePlugin::has_blocks;
    unresponsive.flush = &UnresponsivePlugin::flush;
    unresponsive.destroy = &UnresponsivePlugin::destroy;

    {
        AbiKVCacheStore store(&unresponsive, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                              {file_store.provider->capabilities, std::nullopt}, /*io_timeout_ms=*/20);
        ASSERT_TRUE(store.uses_async_io());

        std::vector<std::vector<uint8_t>> data;
        EXPECT_EQ(read_many(store, {1}, data), (std::vector<bool>{false}));
        EXPECT_TRUE(data[0].empty());
        EXPECT_TRUE(store.is_given_up());
        EXPECT_EQ(UnresponsivePlugin::cancels, 1u);
        EXPECT_EQ(UnresponsivePlugin::waits, 2u) << "one wait that times out, then one for the cancelled token";

        // Nothing reaches the plugin any more, and nothing throws.
        const size_t probes_before = UnresponsivePlugin::probes;
        EXPECT_FALSE(has_one(store, 1));
        EXPECT_EQ(has_many(store, {1, 2}), (std::vector<bool>{false, false}));
        EXPECT_EQ(write_many(store, {1}, {&data[0]}), (std::vector<bool>{false}));
        EXPECT_EQ(read_many(store, {1}, data), (std::vector<bool>{false}));
        EXPECT_NO_THROW(drop_one(store, 1));
        EXPECT_NO_THROW(store.flush());
        EXPECT_EQ(store.get_num_free_blocks(), 0u);
        EXPECT_EQ(UnresponsivePlugin::probes, probes_before);
        EXPECT_EQ(UnresponsivePlugin::submitted, 1u);
    }

    EXPECT_EQ(UnresponsivePlugin::flushes, 0u) << "a store the plugin still uses is not flushed";
    EXPECT_EQ(UnresponsivePlugin::destroys, 0u) << "a store the plugin still uses is not destroyed";

    // The plugin may finish the batch long after the host stopped waiting, so its buffer has to be still there.
    ASSERT_NE(UnresponsivePlugin::held_buffer, nullptr);
    std::memset(UnresponsivePlugin::held_buffer, 0xAB, UnresponsivePlugin::held_bytes);

    file_store.opened.vtable->destroy(file_store.opened.store);
}

TEST(TestAbiFileStoreProvider, AnUnresponsiveWriteIsAMissToo) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    UnresponsivePlugin::reset();
    UnresponsivePlugin::inner = file_store.opened.vtable;
    OvKVCacheStoreVTable unresponsive = *file_store.opened.vtable;
    unresponsive.write_blocks_async = &UnresponsivePlugin::write_async;
    unresponsive.wait = &UnresponsivePlugin::wait;
    unresponsive.cancel = &UnresponsivePlugin::cancel;
    unresponsive.destroy = &UnresponsivePlugin::destroy;

    const std::vector<uint8_t> block = make_block(5);
    {
        AbiKVCacheStore store(&unresponsive, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                              {file_store.provider->capabilities, std::nullopt}, /*io_timeout_ms=*/20);
        EXPECT_EQ(write_many(store, {1}, {&block}), (std::vector<bool>{false}));
        EXPECT_TRUE(store.is_given_up());
    }

    EXPECT_EQ(UnresponsivePlugin::destroys, 0u);
    // The plugin reads a copy that outlives the caller's block, not the caller's block itself.
    EXPECT_NE(UnresponsivePlugin::held_source, block.data());
    ASSERT_NE(UnresponsivePlugin::held_source, nullptr);
    EXPECT_EQ(std::memcmp(UnresponsivePlugin::held_source, block.data(), BLOCK_SIZE), 0);

    file_store.opened.vtable->destroy(file_store.opened.store);
}

TEST(TestAbiFileStoreProvider, AsyncIoIsNotUsedWhenThePluginHasNoCancelEntry) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    OvKVCacheStoreVTable without_cancel = *file_store.opened.vtable;
    without_cancel.cancel = nullptr;

    AbiKVCacheStore store(&without_cancel, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                          {file_store.provider->capabilities, std::nullopt});
    EXPECT_FALSE(store.uses_async_io());
    ASSERT_TRUE(write_one(store, 1, make_block(1)));
    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_one(store, 1, actual));
    EXPECT_EQ(actual, make_block(1));
}

TEST(TestAbiFileStoreProvider, KeysDifferingOnlyInTheHighHalfStayDistinct) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    auto make_key = [](uint64_t hi) {
        OvKVCacheBlockKey key{};
        key.hash = OvKVCacheContentHash{/*lo=*/7, hi};
        return key;
    };
    const auto first_payload = make_block(1);
    const auto second_payload = make_block(2);
    OvKVCacheStoreWriteRequest writes[2]{};
    writes[0].key = make_key(100);
    writes[0].source = first_payload.data();
    writes[0].bytes = first_payload.size();
    writes[1].key = make_key(200);
    writes[1].source = second_payload.data();
    writes[1].bytes = second_payload.size();
    OvKVCacheStoreWriteResult write_results[2]{};
    ASSERT_EQ(opened.vtable->write_blocks(opened.store, writes, write_results, 2), OV_KVCACHE_OK);

    OvKVCacheStoreStats stats{};
    ASSERT_EQ(opened.vtable->stats(opened.store, &stats), OV_KVCACHE_OK);
    EXPECT_EQ(stats.blocks, 2u) << "two keys with the same low half must occupy two blocks";

    for (const auto& [hi, expected] : {std::pair<uint64_t, std::vector<uint8_t>>{100, first_payload},
                                       std::pair<uint64_t, std::vector<uint8_t>>{200, second_payload}}) {
        std::vector<uint8_t> actual(BLOCK_SIZE);
        OvKVCacheStoreReadRequest read{};
        read.key = make_key(hi);
        read.target.kind = OV_KVCACHE_TARGET_HOST;
        read.target.base = actual.data();
        read.target.bytes = actual.size();
        OvKVCacheStoreReadResult read_result{};
        ASSERT_EQ(opened.vtable->read_blocks(opened.store, &read, &read_result, 1), OV_KVCACHE_OK);
        EXPECT_EQ(actual, expected) << "hi=" << hi;
    }
    opened.vtable->destroy(opened.store);
}

TEST(TestAbiFileStoreProvider, BlocksOfEqualHashButDifferentDigestsAreDistinctKeys) {
    auto store = open_store(/*num_blocks=*/4);
    const PrefixKey first{7, ContentHash{1, 100}};
    const PrefixKey second{7, ContentHash{1, 200}};
    const auto first_payload = make_block(1);
    const auto second_payload = make_block(2);

    EXPECT_EQ(store->write_blocks({first, second}, {&first_payload, &second_payload}),
              (std::vector<bool>{true, true}));

    EXPECT_EQ(store->has_blocks({first, second}), (std::vector<bool>{true, true}));
    EXPECT_FALSE(has_one(*store, 7)) << "the hash alone names no block; the digest does";

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(store->read_blocks({second, first}, data), (std::vector<bool>{true, true}));
    EXPECT_EQ(data[0], second_payload);
    EXPECT_EQ(data[1], first_payload);

    store->invalidate(first);
    EXPECT_EQ(store->has_blocks({first, second}), (std::vector<bool>{false, true}));
}

TEST(TestAbiFileStoreProvider, ABlockWithoutADigestIsNeitherStoredNorFound) {
    auto store = open_store(/*num_blocks=*/4);
    const auto payload = make_block(3);

    EXPECT_EQ(store->write_blocks({PrefixKey{42, std::nullopt}}, {&payload}), (std::vector<bool>{false}));
    ASSERT_TRUE(write_one(*store, 42, payload));

    // A block with a digest of the same hash does not make the block without one findable, nor the other way round.
    EXPECT_EQ(store->has_blocks({PrefixKey{42, std::nullopt}}), (std::vector<bool>{false}));
    EXPECT_EQ(store->has_blocks({PrefixKey{42, ContentHash{42, 42}}}), (std::vector<bool>{true}));
    EXPECT_EQ(store->has_blocks({PrefixKey{42, ContentHash{42, 43}}}), (std::vector<bool>{false}));

    // The chain a plugin is asked about ends where a digest is missing.
    EXPECT_EQ(store->has_blocks({PrefixKey{42, ContentHash{42, 42}}, PrefixKey{43, std::nullopt}, PrefixKey{44, ContentHash{44, 44}}}),
              (std::vector<bool>{true, false, false}));
}

namespace {

/// Serves a read of one key from the block of another one, the way a store that mixes up keys would.
struct RedirectHighHalfPlugin {
    static inline const OvKVCacheStoreVTable* inner = nullptr;
    static inline uint64_t from_hi = 0;
    static inline uint64_t to_hi = 0;

    static OvKVCacheStatus read_blocks(void* self, const OvKVCacheStoreReadRequest* r, OvKVCacheStoreReadResult* o, size_t n) {
        std::vector<OvKVCacheStoreReadRequest> redirected(r, r + n);
        for (auto& request : redirected) {
            if (request.key.hash.hi == from_hi) {
                request.key.hash.hi = to_hi;
            }
        }
        return inner->read_blocks(self, redirected.data(), o, n);
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, TheWitnessBindsABlockToBothHalvesOfItsKey) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    RedirectHighHalfPlugin::inner = opened.vtable;
    RedirectHighHalfPlugin::from_hi = 200;
    RedirectHighHalfPlugin::to_hi = 100;
    OvKVCacheStoreVTable redirecting = *opened.vtable;
    redirecting.read_blocks = &RedirectHighHalfPlugin::read_blocks;

    AbiKVCacheStore store(&redirecting, opened.store, "kv", BLOCK_SIZE, 4, AbiKVCacheStore::StoreInfo{});
    const PrefixKey stored{7, ContentHash{1, 100}};
    const PrefixKey requested{7, ContentHash{1, 200}};
    const auto payload = make_block(5);
    ASSERT_EQ(store.write_blocks({stored}, {&payload}), (std::vector<bool>{true}));

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(store.read_blocks({stored}, data), (std::vector<bool>{true}));
    EXPECT_EQ(data[0], payload);
    EXPECT_EQ(store.read_blocks({requested}, data), (std::vector<bool>{false}))
        << "a block served for the other key must fail its witness even though the low halves agree";
}

namespace {
std::vector<std::pair<OvKVCacheLogLevel, std::string>> g_captured_logs;
void capture_log(OvKVCacheLogLevel level, const char* message) {
    g_captured_logs.emplace_back(level, message);
}
}  // namespace

TEST(TestAbiFileStoreProvider, OpenAcceptsMatchingGroupInfoAndForwardsLogsToTheHostCallback) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    const std::string group = "kv";
    const OvKVCacheGroupInfo groups[] = {{group.c_str(), static_cast<uint32_t>(group.size()), 32, BLOCK_SIZE}};
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(group.data()), group.size()};

    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    options.groups = groups;
    options.group_count = 1;
    options.witness_bytes_max = 16;
    options.log = &capture_log;

    g_captured_logs.clear();
    OvKVCacheStoreOpenResult result{};
    ASSERT_EQ(provider->vtable->open(provider->self, &ns, &options, &result), OV_KVCACHE_OK);
    result.vtable->destroy(result.store);
    ASSERT_FALSE(g_captured_logs.empty());
    EXPECT_EQ(g_captured_logs.front().first, static_cast<OvKVCacheLogLevel>(OV_KVCACHE_LOG_DEBUG));
}

TEST(TestAbiFileStoreProvider, OpenRejectsGroupInfoWhoseBlockBytesDisagreeWithTheStoreBlockSize) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    const std::string group = "kv";
    const OvKVCacheGroupInfo groups[] = {{group.c_str(), static_cast<uint32_t>(group.size()), 32, BLOCK_SIZE + 1}};
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(group.data()), group.size()};

    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    options.groups = groups;
    options.group_count = 1;

    OvKVCacheStoreOpenResult result{};
    EXPECT_EQ(provider->vtable->open(provider->self, &ns, &options, &result), OV_KVCACHE_ERR_INVALID_ARG);
    EXPECT_EQ(result.store, nullptr);
}

TEST(TestAbiFileStoreProvider, MissLeavesTheCallersBufferUntouched) {
    auto store = open_store(/*num_blocks=*/4);
    std::vector<uint8_t> data = {1, 2, 3};
    EXPECT_FALSE(read_one(*store, 1234, data));
    EXPECT_EQ(data, (std::vector<uint8_t>{1, 2, 3}));
}

TEST(TestAbiFileStoreProvider, MalformedBlockSizeIsReportedAsInvalidArgInsteadOfThrowing) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    for (const char* bad_value : {"not-a-number", "99999999999999999999999999"}) {
        const OvKVCacheProperty properties[] = {{"block_size_bytes", bad_value}};
        OvKVCacheStoreOpenOptions options{};
        options.capacity_bytes = BLOCK_SIZE * 4;
        options.properties = properties;
        options.property_count = 1;

        OvKVCacheStoreOpenResult result{};
        OvKVCacheStatus status = OV_KVCACHE_OK;
        EXPECT_NO_THROW(status = provider->vtable->open(provider->self, nullptr, &options, &result)) << bad_value;
        EXPECT_EQ(status, OV_KVCACHE_ERR_INVALID_ARG) << bad_value;
        EXPECT_EQ(result.store, nullptr) << bad_value;
    }
}

TEST(TestAbiFileStoreProvider, DropBlockRemovesTheEntry) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    ASSERT_TRUE(has_one(*store, 1));

    drop_one(*store, 1);

    EXPECT_FALSE(has_one(*store, 1));
    std::vector<uint8_t> data;
    EXPECT_FALSE(read_one(*store, 1, data));
}

TEST(TestAbiFileStoreProvider, WriteMakesRoomByEvictingTheLeastRecentlyUsedBlockWhenTheStoreIsFull) {
    auto store = open_store(/*num_blocks=*/2);
    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    ASSERT_TRUE(write_one(*store, 2, make_block(2)));
    EXPECT_EQ(store->get_num_free_blocks(), 0u);

    ASSERT_TRUE(write_one(*store, 3, make_block(3)));

    EXPECT_EQ(has_many(*store, {1, 2, 3}), (std::vector<bool>{false, true, true}));
}

TEST(TestAbiFileStoreProvider, OverwritingAnExistingKeyReplacesItsContents) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(write_one(*store, 1, make_block(1)));
    ASSERT_TRUE(write_one(*store, 1, make_block(9)));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_one(*store, 1, actual));
    EXPECT_EQ(actual, make_block(9));
}

TEST(TestAbiFileStoreProvider, WriteBlocksStoresTheWholeBatchInOneAsyncSubmission) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    AsyncSpy::inner = opened.vtable;
    AsyncSpy::submitted = AsyncSpy::terminal_waits = AsyncSpy::sync_reads = AsyncSpy::sync_writes = 0;
    AsyncSpy::pending_once = 0;
    OvKVCacheStoreVTable spied = *opened.vtable;
    spied.write_blocks_async = &AsyncSpy::write_async;
    spied.wait = &AsyncSpy::wait;

    AbiKVCacheStore store(&spied, opened.store, "kv", BLOCK_SIZE, 4, {provider->capabilities, std::nullopt});
    const std::vector<uint8_t> first = make_block(1);
    const std::vector<uint8_t> second = make_block(2);
    const std::vector<uint8_t> third = make_block(3);
    EXPECT_EQ(write_many(store, {1, 2, 3}, {&first, &second, &third}), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(AsyncSpy::submitted, 1u);

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(read_many(store, {1, 2, 3}, data), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(data[0], first);
    EXPECT_EQ(data[2], third);
}

TEST(TestAbiFileStoreProvider, WriteBlocksTakesAWholeBatchLargerThanTheStoreAndKeepsTheNewestBlocks) {
    auto store = open_store(/*num_blocks=*/2);
    const std::vector<uint8_t> first = make_block(1);
    const std::vector<uint8_t> second = make_block(2);
    const std::vector<uint8_t> third = make_block(3);

    EXPECT_EQ(write_many(*store, {1, 2, 3}, {&first, &second, &third}), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(has_many(*store, {1, 2, 3}), (std::vector<bool>{false, true, true}));
    EXPECT_TRUE(write_many(*store, {}, {}).empty());
}

namespace {

/// Records the restore reports a plugin gets, in place of the file provider's missing entry.
struct RestoreSpy {
    static inline size_t calls = 0;
    static inline OvKVCacheRestoreStats last{};

    static void on_restore_finish(void* /*self*/, const OvKVCacheRestoreStats* stats) {
        ++calls;
        last = *stats;
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, FileStoreTakesNoRestoreReports) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    EXPECT_EQ(provider->capabilities & OV_KVCACHE_CAP_RESTORE_STATS, 0u);
    auto store = open_store(/*num_blocks=*/2);
    EXPECT_FALSE(store->takes_restore_stats());
    EXPECT_FALSE(store->report_restore_stats(RestoreStats{100, 10, 20, 30}));
}

TEST(TestAbiFileStoreProvider, RestoreStatsReachThePluginWhenItAdvertisesTheCapability) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    RestoreSpy::calls = 0;
    OvKVCacheStoreVTable reporting = *file_store.opened.vtable;
    reporting.on_restore_finish = &RestoreSpy::on_restore_finish;

    AbiKVCacheStore store(&reporting, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                          {file_store.provider->capabilities | OV_KVCACHE_CAP_RESTORE_STATS, std::nullopt});
    ASSERT_TRUE(store.takes_restore_stats());

    EXPECT_TRUE(store.report_restore_stats(RestoreStats{16384, 1024, 4096, 5000}));
    EXPECT_EQ(RestoreSpy::calls, 1u);
    EXPECT_EQ(RestoreSpy::last.struct_bytes, sizeof(OvKVCacheRestoreStats));
    EXPECT_EQ(RestoreSpy::last.prompt_tokens, 16384u);
    EXPECT_EQ(RestoreSpy::last.device_hit_tokens, 1024u);
    EXPECT_EQ(RestoreSpy::last.store_hit_tokens, 4096u);
    EXPECT_EQ(RestoreSpy::last.reused_tokens, 5000u);
}

TEST(TestAbiFileStoreProvider, RestoreStatsAreNotSentWithoutTheCapabilityOrTheEntry) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    RestoreSpy::calls = 0;
    OvKVCacheStoreVTable reporting = *file_store.opened.vtable;
    reporting.on_restore_finish = &RestoreSpy::on_restore_finish;

    // The entry is there but the capability was not passed: the host does not use an entry it was not told about.
    AbiKVCacheStore without_capability(&reporting, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                                       {file_store.provider->capabilities, std::nullopt});
    EXPECT_FALSE(without_capability.report_restore_stats(RestoreStats{1, 2, 3, 4}));
    EXPECT_EQ(RestoreSpy::calls, 0u);

    // The capability is advertised but the entry is not set. A store instance is destroyed once, so this is a new one.
    auto other_file_store = open_file_store(/*num_blocks=*/4);
    OvKVCacheStoreVTable without_entry = *other_file_store.opened.vtable;
    without_entry.on_restore_finish = nullptr;
    AbiKVCacheStore without_function(&without_entry, other_file_store.opened.store, "kv", BLOCK_SIZE, 4,
                                     {other_file_store.provider->capabilities | OV_KVCACHE_CAP_RESTORE_STATS,
                                      std::nullopt});
    EXPECT_FALSE(without_function.takes_restore_stats());
    EXPECT_FALSE(without_function.report_restore_stats(RestoreStats{1, 2, 3, 4}));
    EXPECT_EQ(RestoreSpy::calls, 0u);
}

TEST(TestAbiFileStoreProvider, RestoreStatsAreNotSentThroughAVtableTooShortToHoldTheEntry) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    RestoreSpy::calls = 0;
    OvKVCacheStoreVTable short_vtable = *file_store.opened.vtable;
    short_vtable.on_restore_finish = &RestoreSpy::on_restore_finish;
    // An ABI 1.1 plugin builds a vtable that ends at `cancel`.
    short_vtable.struct_bytes = static_cast<uint32_t>(offsetof(OvKVCacheStoreVTable, on_restore_finish));

    AbiKVCacheStore store(&short_vtable, file_store.opened.store, "kv", BLOCK_SIZE, 4,
                          {file_store.provider->capabilities | OV_KVCACHE_CAP_RESTORE_STATS, std::nullopt});
    EXPECT_FALSE(store.takes_restore_stats());
    EXPECT_FALSE(store.report_restore_stats(RestoreStats{1, 2, 3, 4}));
    EXPECT_EQ(RestoreSpy::calls, 0u);
}

TEST(TestAbiFileStoreProvider, RestoreStatsNameTheGroupOfTheView) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    RestoreSpy::calls = 0;
    OvKVCacheStoreVTable reporting = *file_store.opened.vtable;
    reporting.on_restore_finish = &RestoreSpy::on_restore_finish;

    AbiKVCacheStore store(&reporting, file_store.opened.store, "la", BLOCK_SIZE, 4,
                          {file_store.provider->capabilities | OV_KVCACHE_CAP_RESTORE_STATS, std::nullopt});
    ASSERT_TRUE(store.report_restore_stats(RestoreStats{1, 2, 3, 4}));
    EXPECT_EQ(std::string(RestoreSpy::last.group, RestoreSpy::last.group_len), "la");
}

namespace {

/// Records what a plugin is told about the tokens of linear-attention checkpoints.
struct PrefixTokensSpy {
    static inline size_t calls = 0;
    static inline std::vector<OvKVCachePrefixTokens> items;
    static inline std::vector<std::vector<int64_t>> token_copies;

    static void reset() {
        calls = 0;
        items.clear();
        token_copies.clear();
    }

    static OvKVCacheStatus announce(void* /*self*/, const OvKVCachePrefixTokens* announced, size_t count) {
        ++calls;
        for (size_t i = 0; i < count; ++i) {
            items.push_back(announced[i]);
            token_copies.emplace_back(announced[i].token_ids, announced[i].token_ids + announced[i].token_count);
        }
        return OV_KVCACHE_OK;
    }
};

std::vector<PrefixTokens> make_prefix_tokens(const ContentHash& digest, std::vector<int64_t> token_ids) {
    return {PrefixTokens{PrefixKey{1, digest}, std::make_shared<const std::vector<int64_t>>(std::move(token_ids))}};
}

}  // namespace

TEST(TestAbiFileStoreProvider, PrefixTokensReachThePluginWhenItAdvertisesTheCapability) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    PrefixTokensSpy::reset();
    OvKVCacheStoreVTable announcing = *file_store.opened.vtable;
    announcing.announce_prefix_tokens = &PrefixTokensSpy::announce;

    AbiKVCacheStore store(&announcing, file_store.opened.store, "la", BLOCK_SIZE, 4,
                          {file_store.provider->capabilities | OV_KVCACHE_CAP_PREFIX_TOKENS, std::nullopt});
    ASSERT_TRUE(store.wants_prefix_tokens());

    const ContentHash digest{0x1122334455667788ull, 0x99aabbccddeeff00ull};
    store.announce_prefix_tokens(make_prefix_tokens(digest, {5, 6, 7, 8}));

    ASSERT_EQ(PrefixTokensSpy::calls, 1u);
    ASSERT_EQ(PrefixTokensSpy::items.size(), 1u);
    EXPECT_EQ(PrefixTokensSpy::items[0].hash.lo, digest.lo);
    EXPECT_EQ(PrefixTokensSpy::items[0].hash.hi, digest.hi);
    EXPECT_EQ(PrefixTokensSpy::items[0].token_count, 4u);
    EXPECT_EQ(PrefixTokensSpy::token_copies[0], (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestAbiFileStoreProvider, PrefixTokensAreNotSentWithoutTheCapabilityOrTheEntry) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    PrefixTokensSpy::reset();
    OvKVCacheStoreVTable announcing = *file_store.opened.vtable;
    announcing.announce_prefix_tokens = &PrefixTokensSpy::announce;
    const ContentHash digest{1, 2};

    // The entry is there but the capability was not passed: token ids are user content, so they stay home.
    AbiKVCacheStore without_capability(&announcing, file_store.opened.store, "la", BLOCK_SIZE, 4,
                                       {file_store.provider->capabilities, std::nullopt});
    EXPECT_FALSE(without_capability.wants_prefix_tokens());
    without_capability.announce_prefix_tokens(make_prefix_tokens(digest, {1}));
    EXPECT_EQ(PrefixTokensSpy::calls, 0u);

    // The capability is advertised but the entry is not set.
    auto other_file_store = open_file_store(/*num_blocks=*/4);
    OvKVCacheStoreVTable without_entry = *other_file_store.opened.vtable;
    without_entry.announce_prefix_tokens = nullptr;
    AbiKVCacheStore without_function(&without_entry, other_file_store.opened.store, "la", BLOCK_SIZE, 4,
                                     {other_file_store.provider->capabilities | OV_KVCACHE_CAP_PREFIX_TOKENS,
                                      std::nullopt});
    EXPECT_FALSE(without_function.wants_prefix_tokens());

    // An ABI 1.2 plugin builds a vtable that ends before the entry.
    auto third_file_store = open_file_store(/*num_blocks=*/4);
    OvKVCacheStoreVTable short_vtable = *third_file_store.opened.vtable;
    short_vtable.announce_prefix_tokens = &PrefixTokensSpy::announce;
    short_vtable.struct_bytes = static_cast<uint32_t>(offsetof(OvKVCacheStoreVTable, announce_prefix_tokens));
    AbiKVCacheStore too_short(&short_vtable, third_file_store.opened.store, "la", BLOCK_SIZE, 4,
                              {third_file_store.provider->capabilities | OV_KVCACHE_CAP_PREFIX_TOKENS, std::nullopt});
    EXPECT_FALSE(too_short.wants_prefix_tokens());
    too_short.announce_prefix_tokens(make_prefix_tokens(digest, {1}));
    EXPECT_EQ(PrefixTokensSpy::calls, 0u);
}

namespace {

struct DestroySpy {
    static inline size_t calls = 0;
    static inline void (*real_destroy)(void*) = nullptr;

    static void destroy(void* self) {
        ++calls;
        real_destroy(self);
    }
};

}  // namespace

TEST(TestAbiFileStoreProvider, ViewsOfOneInstanceShareItsLifetimeAndAreEachUsable) {
    auto file_store = open_file_store(/*num_blocks=*/4);
    DestroySpy::calls = 0;
    DestroySpy::real_destroy = file_store.opened.vtable->destroy;
    OvKVCacheStoreVTable counting = *file_store.opened.vtable;
    counting.destroy = &DestroySpy::destroy;

    auto instance = std::make_shared<AbiStoreInstance>(&counting, file_store.opened.store);
    const AbiKVCacheStore::StoreInfo info{file_store.provider->capabilities, std::nullopt};
    auto kv_view = std::make_unique<AbiKVCacheStore>(instance, "kv", BLOCK_SIZE, 4, info);
    auto la_view = std::make_unique<AbiKVCacheStore>(instance, "la", BLOCK_SIZE, 4, info);
    instance.reset();

    ASSERT_TRUE(write_one(*kv_view, 7, make_block(1)));
    EXPECT_TRUE(has_one(*kv_view, 7));
    ASSERT_TRUE(write_one(*la_view, 8, make_block(2)));

    // The instance outlives the view that was first closed.
    kv_view.reset();
    EXPECT_EQ(DestroySpy::calls, 0u);
    std::vector<uint8_t> data;
    ASSERT_TRUE(read_one(*la_view, 8, data));
    EXPECT_EQ(data, make_block(2));

    la_view.reset();
    EXPECT_EQ(DestroySpy::calls, 1u);
}

TEST(TestAbiFileStoreProvider, ConcurrentWritesReadsAndDropsNeverExposeAHalfPublishedBlock) {
    constexpr size_t num_keys = 8;
    constexpr size_t rounds = 300;
    auto store = open_store(/*num_blocks=*/num_keys);
    std::atomic<size_t> failures{0};

    std::thread writer([&] {
        for (size_t round = 0; round < rounds; ++round) {
            for (size_t key = 0; key < num_keys; ++key) {
                try {
                    write_one(*store, key, make_block(static_cast<uint8_t>(key)));
                    drop_one(*store, (key + 3) % num_keys);
                } catch (const std::exception&) {
                    ++failures;
                }
            }
        }
    });
    std::thread reader([&] {
        for (size_t round = 0; round < rounds; ++round) {
            for (size_t key = 0; key < num_keys; ++key) {
                try {
                    std::vector<uint8_t> data;
                    if (read_one(*store, key, data) && data != make_block(static_cast<uint8_t>(key))) {
                        ++failures;
                    }
                } catch (const std::exception&) {
                    ++failures;
                }
            }
        }
    });
    writer.join();
    reader.join();

    EXPECT_EQ(failures.load(), 0u);
}
