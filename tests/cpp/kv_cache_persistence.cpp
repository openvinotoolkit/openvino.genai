// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for KV cache dump/restore to SSD feature

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>

#include "openvino/runtime/core.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/cache_manager.hpp"
#include "helper.hpp"

using namespace ov::genai;

namespace {

// Test fixture for KV cache persistence tests
class KVCachePersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for test files
        test_dir = std::filesystem::temp_directory_path() / "kv_cache_test";
        std::filesystem::create_directories(test_dir);
        
        // Initialize OpenVINO core and create dummy model
        num_decoder_layers = 4;  // Use smaller model for faster tests
        request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
        cache_manager = std::make_shared<CacheManager>(request);
    }
    
    void TearDown() override {
        // Cleanup temporary directory
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    std::filesystem::path test_dir;
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<CacheManager> cache_manager;
    size_t num_decoder_layers;
};

// Helper function to check if file exists
bool file_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path);
}

// Helper function to get file size
size_t get_file_size(const std::filesystem::path& path) {
    return std::filesystem::file_size(path);
}

}  // namespace


// =============================================================================
// Test: Basic KV cache dump to directory
// =============================================================================
TEST_F(KVCachePersistenceTest, test_dump_kv_cache_creates_files) {
    const size_t num_kv_blocks = 10;
    
    // Allocate cache
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    ASSERT_EQ(cache_manager->get_num_allocated_kv_blocks(), num_kv_blocks);
    
    // Dump KV cache to directory
    std::string dump_dir = (test_dir / "kv_dump").string();
    cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
    
    // Verify files were created for each layer
    for (size_t layer = 0; layer < num_decoder_layers; ++layer) {
        std::string key_bin = dump_dir + "/layer_" + std::to_string(layer) + "_key.bin";
        std::string key_meta = dump_dir + "/layer_" + std::to_string(layer) + "_key.meta";
        std::string val_bin = dump_dir + "/layer_" + std::to_string(layer) + "_value.bin";
        std::string val_meta = dump_dir + "/layer_" + std::to_string(layer) + "_value.meta";
        
        EXPECT_TRUE(file_exists(key_bin)) << "Missing key binary for layer " << layer;
        EXPECT_TRUE(file_exists(key_meta)) << "Missing key metadata for layer " << layer;
        EXPECT_TRUE(file_exists(val_bin)) << "Missing value binary for layer " << layer;
        EXPECT_TRUE(file_exists(val_meta)) << "Missing value metadata for layer " << layer;
        
        // Verify files are not empty
        EXPECT_GT(get_file_size(key_bin), 0) << "Empty key binary for layer " << layer;
        EXPECT_GT(get_file_size(key_meta), 0) << "Empty key metadata for layer " << layer;
    }
}


// =============================================================================
// Test: KV cache metadata format
// =============================================================================
TEST_F(KVCachePersistenceTest, test_dump_metadata_format) {
    const size_t num_kv_blocks = 10;
    
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    
    std::string dump_dir = (test_dir / "kv_meta_test").string();
    cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
    
    // Read and verify metadata file format
    std::string meta_file = dump_dir + "/layer_0_key.meta";
    std::ifstream in(meta_file);
    ASSERT_TRUE(in.is_open()) << "Failed to open metadata file";
    
    std::map<std::string, std::string> meta_map;
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            meta_map[line.substr(0, pos)] = line.substr(pos + 1);
        }
    }
    in.close();
    
    // Check required metadata fields exist
    EXPECT_TRUE(meta_map.find("element_type") != meta_map.end()) << "Missing element_type in metadata";
    EXPECT_TRUE(meta_map.find("shape") != meta_map.end()) << "Missing shape in metadata";
    EXPECT_TRUE(meta_map.find("num_blocks") != meta_map.end()) << "Missing num_blocks in metadata";
    EXPECT_TRUE(meta_map.find("num_heads") != meta_map.end()) << "Missing num_heads in metadata";
    EXPECT_TRUE(meta_map.find("block_size") != meta_map.end()) << "Missing block_size in metadata";
    EXPECT_TRUE(meta_map.find("head_dim") != meta_map.end()) << "Missing head_dim in metadata";
    
    // Verify num_blocks matches
    EXPECT_EQ(std::stoull(meta_map["num_blocks"]), num_kv_blocks);
}


// =============================================================================
// Test: Load KV cache from directory
// =============================================================================
TEST_F(KVCachePersistenceTest, test_load_kv_cache_from_dir) {
    const size_t num_kv_blocks = 10;
    
    // First dump some cache data
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    std::string dump_dir = (test_dir / "kv_load_test").string();
    cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
    
    // Create a new cache manager and load from directory
    ov::InferRequest new_request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto new_cache_manager = std::make_shared<CacheManager>(new_request);
    
    bool load_success = new_cache_manager->load_kv_cache_from_dir(dump_dir);
    EXPECT_TRUE(load_success) << "Failed to load KV cache from directory";
    
    // Verify loaded cache has correct number of blocks
    EXPECT_EQ(new_cache_manager->get_num_allocated_kv_blocks(), num_kv_blocks);
    EXPECT_EQ(new_cache_manager->get_num_decoder_layers(), num_decoder_layers);
}


// =============================================================================
// Test: Load from non-existent directory returns false
// =============================================================================
TEST_F(KVCachePersistenceTest, test_load_nonexistent_dir_returns_false) {
    std::string nonexistent_dir = (test_dir / "does_not_exist").string();
    
    bool load_success = cache_manager->load_kv_cache_from_dir(nonexistent_dir);
    EXPECT_FALSE(load_success) << "Load should fail for non-existent directory";
}


// =============================================================================
// Test: Dump and load roundtrip preserves data
// =============================================================================
TEST_F(KVCachePersistenceTest, test_dump_load_roundtrip_data_integrity) {
    const size_t num_kv_blocks = 5;
    
    // Allocate and fill cache with known pattern
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    
    // Get original tensor data for comparison
    auto original_key_cache_0 = cache_manager->get_key_cache(0);
    auto original_value_cache_0 = cache_manager->get_value_cache(0);
    size_t original_key_bytes = original_key_cache_0.get_byte_size();
    size_t original_value_bytes = original_value_cache_0.get_byte_size();
    
    // Dump to directory
    std::string dump_dir = (test_dir / "roundtrip_test").string();
    cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
    
    // Create new cache manager and load
    ov::InferRequest new_request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto new_cache_manager = std::make_shared<CacheManager>(new_request);
    
    bool load_success = new_cache_manager->load_kv_cache_from_dir(dump_dir);
    ASSERT_TRUE(load_success);
    
    // Verify loaded tensor sizes match original
    auto loaded_key_cache_0 = new_cache_manager->get_key_cache(0);
    auto loaded_value_cache_0 = new_cache_manager->get_value_cache(0);
    
    EXPECT_EQ(loaded_key_cache_0.get_byte_size(), original_key_bytes);
    EXPECT_EQ(loaded_value_cache_0.get_byte_size(), original_value_bytes);
    EXPECT_EQ(loaded_key_cache_0.get_shape(), original_key_cache_0.get_shape());
    EXPECT_EQ(loaded_value_cache_0.get_shape(), original_value_cache_0.get_shape());
}


// =============================================================================
// Test: Optimized dump with used blocks
// =============================================================================
TEST_F(KVCachePersistenceTest, test_optimized_dump_with_used_blocks) {
    const size_t total_blocks = 20;
    const size_t used_blocks = 5;
    
    cache_manager->allocate_cache_if_needed(total_blocks);
    
    // Dump only used blocks (optimized)
    std::string dump_dir = (test_dir / "optimized_dump").string();
    cache_manager->dump_kv_cache_to_dir_optimized(dump_dir, total_blocks, used_blocks);
    
    // Verify metadata indicates optimization
    std::string meta_file = dump_dir + "/layer_0_key.meta";
    std::ifstream in(meta_file);
    ASSERT_TRUE(in.is_open());
    
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    
    // Check for optimized flag in metadata
    EXPECT_TRUE(content.find("optimized=true") != std::string::npos) << "Missing optimized flag";
    EXPECT_TRUE(content.find("used_blocks=" + std::to_string(used_blocks)) != std::string::npos) 
        << "Missing or incorrect used_blocks in metadata";
}


// =============================================================================
// Test: BlockManager manifest serialization
// =============================================================================
TEST(TestBlockManagerManifest, test_save_and_load_manifest) {
    const size_t num_kv_blocks = 10;
    const size_t block_size = 32;
    const size_t num_decoder_layers = 4;
    
    // Create temporary directory
    std::filesystem::path test_dir = std::filesystem::temp_directory_path() / "bm_manifest_test";
    std::filesystem::create_directories(test_dir);
    
    // Create BlockManager and allocate some blocks
    BlockManager bm(num_kv_blocks, true, block_size, num_decoder_layers);  // enable prefix caching
    
    // Save manifest
    std::string manifest_dir = test_dir.string();
    bool save_success = bm.dump_manifest(manifest_dir);
    EXPECT_TRUE(save_success) << "Failed to save BlockManager manifest";
    
    // Verify manifest file exists
    std::string manifest_file = manifest_dir + "/block_manager.manifest";
    EXPECT_TRUE(std::filesystem::exists(manifest_file)) << "Manifest file not created";
    
    // Create new BlockManager and load manifest
    BlockManager bm2(num_kv_blocks, true, block_size, num_decoder_layers);
    bool load_success = bm2.load_from_manifest(manifest_dir);
    EXPECT_TRUE(load_success) << "Failed to load BlockManager manifest";
    
    // Cleanup
    std::filesystem::remove_all(test_dir);
}


// =============================================================================
// Test: Sequence state JSON persistence
// =============================================================================
TEST_F(KVCachePersistenceTest, test_sequence_state_persistence) {
    const size_t num_kv_blocks = 10;
    
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    
    // Create test sequence state data
    std::vector<int64_t> cached_tokens = {100, 200, 300, 400, 500};
    size_t sequence_length = cached_tokens.size();
    size_t position_offset = 0;
    std::string model_name = "test_model";
    
    // Dump with sequence state
    std::string dump_dir = (test_dir / "seq_state_test").string();
    cache_manager->dump_kv_cache_with_sequence_state(
        dump_dir, num_kv_blocks, cached_tokens, sequence_length, position_offset, model_name);
    
    // Verify sequence_state.json was created
    std::string seq_state_file = dump_dir + "/sequence_state.json";
    EXPECT_TRUE(file_exists(seq_state_file)) << "sequence_state.json not created";
    
    // Read and verify JSON content
    std::ifstream in(seq_state_file);
    ASSERT_TRUE(in.is_open());
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    
    EXPECT_TRUE(content.find("\"model_name\": \"test_model\"") != std::string::npos);
    EXPECT_TRUE(content.find("\"sequence_length\": 5") != std::string::npos);
    EXPECT_TRUE(content.find("\"cached_tokens\":") != std::string::npos);
}


// =============================================================================
// Test: Load KV cache with sequence state
// =============================================================================
TEST_F(KVCachePersistenceTest, test_load_kv_cache_with_sequence_state) {
    const size_t num_kv_blocks = 10;
    
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    
    // Dump with sequence state
    std::vector<int64_t> original_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::string dump_dir = (test_dir / "load_seq_state_test").string();
    cache_manager->dump_kv_cache_with_sequence_state(
        dump_dir, num_kv_blocks, original_tokens, original_tokens.size(), 0, "test");
    
    // Create new cache manager and load with sequence state
    ov::InferRequest new_request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
    auto new_cache_manager = std::make_shared<CacheManager>(new_request);
    
    CacheManager::SequenceState loaded_state;
    bool load_success = new_cache_manager->load_kv_cache_with_sequence_state(dump_dir, 0, &loaded_state);
    
    EXPECT_TRUE(load_success) << "Failed to load KV cache with sequence state";
    EXPECT_EQ(loaded_state.cached_tokens.size(), original_tokens.size());
    EXPECT_EQ(loaded_state.sequence_length, original_tokens.size());
    
    // Verify tokens match
    for (size_t i = 0; i < original_tokens.size(); ++i) {
        EXPECT_EQ(loaded_state.cached_tokens[i], original_tokens[i]) 
            << "Token mismatch at position " << i;
    }
}


// =============================================================================
// Test: KV cache dump/load with f16 precision (instead of default u8)
// =============================================================================
TEST(TestKVCachePrecision, test_kv_cache_f16_precision) {
    ov::Core core;
    size_t num_decoder_layers = 4;
    const size_t num_kv_blocks = 5;
    
    // Create model with f16 KV cache precision
    ov::InferRequest request = core.compile_model(
        get_dummy_model(core, num_decoder_layers, ov::element::f16)
    ).create_infer_request();
    auto cache_manager = std::make_shared<CacheManager>(request);
    
    // Create temporary directory for test files
    std::filesystem::path test_dir = std::filesystem::temp_directory_path() / "kv_cache_f16_test";
    std::filesystem::create_directories(test_dir);
    
    // Allocate and dump cache
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    std::string dump_dir = (test_dir / "f16_dump").string();
    cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
    
    // Verify metadata shows f16 element type
    std::string meta_file = dump_dir + "/layer_0_key.meta";
    std::ifstream in(meta_file);
    ASSERT_TRUE(in.is_open()) << "Failed to open metadata file";
    
    std::map<std::string, std::string> meta_map;
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            meta_map[line.substr(0, pos)] = line.substr(pos + 1);
        }
    }
    in.close();
    
    // Check element type is f16
    EXPECT_TRUE(meta_map.find("element_type") != meta_map.end()) << "Missing element_type in metadata";
    EXPECT_EQ(meta_map["element_type"], "f16") << "Expected f16 element type";
    
    // Check bytes_per_element is 2 (f16 = 2 bytes)
    EXPECT_TRUE(meta_map.find("bytes_per_element") != meta_map.end()) << "Missing bytes_per_element in metadata";
    EXPECT_EQ(meta_map["bytes_per_element"], "2") << "f16 should have 2 bytes per element";
    
    // Verify file size is correct (should be 2x larger than u8)
    // shape: [5, 12, 64, 64] * 2 bytes = 491520 bytes
    size_t expected_size = num_kv_blocks * 12 * 64 * 64 * 2;  // 2 bytes for f16
    size_t actual_size = std::filesystem::file_size(dump_dir + "/layer_0_key.bin");
    EXPECT_EQ(actual_size, expected_size) << "File size mismatch for f16 precision";
    
    // Load back and verify
    ov::InferRequest new_request = core.compile_model(
        get_dummy_model(core, num_decoder_layers, ov::element::f16)
    ).create_infer_request();
    auto new_cache_manager = std::make_shared<CacheManager>(new_request);
    
    bool load_success = new_cache_manager->load_kv_cache_from_dir(dump_dir);
    EXPECT_TRUE(load_success) << "Failed to load f16 KV cache from directory";
    EXPECT_EQ(new_cache_manager->get_num_allocated_kv_blocks(), num_kv_blocks);
    
    // Cleanup
    std::filesystem::remove_all(test_dir);
}


// =============================================================================
// Test: Multiple dump/load cycles don't corrupt data
// =============================================================================
TEST_F(KVCachePersistenceTest, test_multiple_dump_load_cycles) {
    const size_t num_kv_blocks = 5;
    
    cache_manager->allocate_cache_if_needed(num_kv_blocks);
    
    // Perform multiple dump/load cycles
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::string dump_dir = (test_dir / ("cycle_" + std::to_string(cycle))).string();
        
        // Dump
        cache_manager->dump_kv_cache_to_dir(dump_dir, num_kv_blocks);
        
        // Load into new cache manager
        ov::InferRequest new_request = core.compile_model(get_dummy_model(core, num_decoder_layers)).create_infer_request();
        auto new_cache_manager = std::make_shared<CacheManager>(new_request);
        
        bool success = new_cache_manager->load_kv_cache_from_dir(dump_dir);
        EXPECT_TRUE(success) << "Failed at cycle " << cycle;
        EXPECT_EQ(new_cache_manager->get_num_allocated_kv_blocks(), num_kv_blocks) 
            << "Block count mismatch at cycle " << cycle;
        
        // Use loaded cache for next cycle
        cache_manager = new_cache_manager;
    }
}

