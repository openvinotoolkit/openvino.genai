// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/genai/json_container.hpp"
#include "json_utils.hpp"

using namespace ov::genai;

namespace {
    constexpr bool BOOL_VALUE = true;
    constexpr int INT_VALUE = 42;
    constexpr int64_t INT64_VALUE = 10000000000;
    constexpr double DOUBLE_VALUE = 3.14;
    constexpr float FLOAT_VALUE = 2.71f;
    const std::string TEST_STRING = "string";
    const char* C_STRING_VALUE = "c_string";
}

TEST(JsonContainerTest, constructors) {
    // Default constructor
    JsonContainer empty_json;
    EXPECT_TRUE(empty_json.is_object());
    EXPECT_EQ(empty_json.size(), 0);
    EXPECT_TRUE(empty_json.empty());

    // Initializer list constructor
    JsonContainer init_list_json({{"key1", TEST_STRING}, {"key2", INT_VALUE}, {"key3", BOOL_VALUE}});
    EXPECT_TRUE(init_list_json.is_object());
    EXPECT_EQ(init_list_json.size(), 3);
    EXPECT_FALSE(init_list_json.empty());
    EXPECT_TRUE(init_list_json.contains("key1"));
    EXPECT_FALSE(init_list_json.contains("nonexistent"));
    EXPECT_EQ(init_list_json["key1"].get_string(), TEST_STRING);
    EXPECT_EQ(init_list_json["key2"].get_int(), INT_VALUE);
    EXPECT_EQ(init_list_json["key3"].get_bool(), BOOL_VALUE);

    // AnyMap constructor
    ov::AnyMap any_map = {{"key1", TEST_STRING}, {"key2", DOUBLE_VALUE}};
    JsonContainer any_map_json(any_map);
    EXPECT_TRUE(any_map_json.is_object());
    EXPECT_EQ(any_map_json.size(), 2);
    EXPECT_EQ(any_map_json["key1"].get_string(), TEST_STRING);
    EXPECT_DOUBLE_EQ(any_map_json["key2"].get_double(), DOUBLE_VALUE);

    // Copy constructor
    JsonContainer original({{"key", "value"}});
    JsonContainer copied(original);
    EXPECT_EQ(copied, original);
    copied["key"] = "modified";
    EXPECT_NE(copied, original);
    EXPECT_EQ(original["key"].get_string(), "value");
    EXPECT_EQ(copied["key"].get_string(), "modified");

    // Move constructor
    JsonContainer source({{"temp", "data"}});
    std::string source_json_string = source.to_json_string();
    JsonContainer target(std::move(source));
    EXPECT_EQ(target["temp"].get_string(), "data");
    EXPECT_EQ(target.to_json_string(), source_json_string);
}

TEST(JsonContainerTest, from_json_string) {
    std::string json_str = R"({"name": "test", "count": 10, "flag": false, "data": [1, 2, 3]})";
    JsonContainer json_from_string = JsonContainer::from_json_string(json_str);
    EXPECT_TRUE(json_from_string.is_object());
    EXPECT_EQ(json_from_string.size(), 4);
    EXPECT_EQ(json_from_string["name"].get_string(), "test");
    EXPECT_EQ(json_from_string["count"].get_int(), 10);
    EXPECT_EQ(json_from_string["flag"].get_bool(), false);
    EXPECT_TRUE(json_from_string["data"].is_array());
    EXPECT_EQ(json_from_string["data"].size(), 3);
    EXPECT_EQ(json_from_string["data"][0].get_int(), 1);
    EXPECT_EQ(json_from_string["data"][1].get_int(), 2);
    EXPECT_EQ(json_from_string["data"][2].get_int(), 3);
    EXPECT_THROW(JsonContainer::from_json_string("invalid json string"), ov::Exception);
}

TEST(JsonContainerTest, equality) {
    JsonContainer original({{"key", "value"}});
    JsonContainer different({{"other", "value"}});
    EXPECT_NE(original, different);
    JsonContainer original_same_structure({{"key", "value"}});
    EXPECT_EQ(original, original_same_structure);
    JsonContainer nested = JsonContainer::from_json_string(R"({"nested": {"key": "value"}})");
    JsonContainer original_same_structure_different_path = nested["nested"];
    EXPECT_EQ(original, original_same_structure_different_path);
}

TEST(JsonContainerTest, copy) {
    JsonContainer source({{"key", "value"}});
    
    // Copy assignment
    JsonContainer copy_target;
    copy_target = source;
    EXPECT_EQ(copy_target, source);
    EXPECT_EQ(copy_target["key"].get_string(), "value");
    copy_target["key"] = "modified";
    copy_target["new_key"] = "added";
    EXPECT_NE(copy_target, source);
    EXPECT_EQ(source["key"].get_string(), "value");
    EXPECT_EQ(copy_target["key"].get_string(), "modified");
    EXPECT_FALSE(source.contains("new_key"));

    // Copy assignment to path
    JsonContainer root({{"section1", JsonContainer({{"value", "old"}})},
                        {"section2", JsonContainer({{"value", "other"}})}});
    JsonContainer other_section({{"value", "new_content"}, {"extra", "data"}});
    root["section1"] = other_section;
    EXPECT_EQ(root["section1"]["value"].get_string(), "new_content");
    EXPECT_EQ(root["section1"]["extra"].get_string(), "data");
    EXPECT_EQ(root["section2"]["value"].get_string(), "other");
    other_section["value"] = "modified_source";
    EXPECT_EQ(root["section1"]["value"].get_string(), "new_content");

    // Copy method (explicit)
    JsonContainer original({{"nested", JsonContainer({{"key", "value"}})}});
    JsonContainer copied = original.copy();
    EXPECT_EQ(copied, original);
    EXPECT_EQ(copied["nested"]["key"].get_string(), "value");
    copied["nested"]["key"] = "modified";
    copied["new_section"] = JsonContainer({{"data", "new"}});
    EXPECT_EQ(original["nested"]["key"].get_string(), "value");
    EXPECT_FALSE(original.contains("new_section"));
    EXPECT_NE(copied, original);
}

TEST(JsonContainerTest, share) {
    JsonContainer original({{"key", "value"}});

    JsonContainer shared = original.share();
    EXPECT_EQ(shared, original);
    EXPECT_EQ(shared["key"].get_string(), "value");
    shared["key"] = "modified";
    shared["new_key"] = "added";
    EXPECT_EQ(original["key"].get_string(), "modified");
    EXPECT_EQ(original["new_key"].get_string(), "added");
    EXPECT_EQ(shared, original);
    JsonContainer other({{"key", "other_value"}});
    shared = other;
    EXPECT_EQ(shared["key"].get_string(), "other_value");
    EXPECT_EQ(original["key"].get_string(), "other_value");
    EXPECT_FALSE(shared.contains("new_key"));
    EXPECT_FALSE(original.contains("new_key"));

    // Sharing chains
    JsonContainer root = JsonContainer::from_json_string(R"({"level1": {"level2": {"data": "original"}}})");
    JsonContainer shared_root = root.share();
    JsonContainer shared_level1 = root["level1"].share();
    JsonContainer shared_level2 = root["level1"]["level2"].share();
    EXPECT_EQ(shared_root["level1"]["level2"]["data"].get_string(), "original");
    EXPECT_EQ(shared_level1["level2"]["data"].get_string(), "original");
    EXPECT_EQ(shared_level2["data"].get_string(), "original");
    shared_level2["data"] = "modified";
    shared_level2["new_data"] = "added";
    EXPECT_EQ(root["level1"]["level2"]["data"].get_string(), "modified");
    EXPECT_EQ(shared_root["level1"]["level2"]["data"].get_string(), "modified");
    EXPECT_EQ(shared_level1["level2"]["data"].get_string(), "modified");
    EXPECT_EQ(shared_level2["new_data"].get_string(), "added");
}

TEST(JsonContainerTest, move_assignment) {
    JsonContainer source({{"key", "value"}});
    JsonContainer move_target({{"old", "data"}});
    move_target = std::move(source);
    EXPECT_EQ(move_target["key"].get_string(), "value");
    EXPECT_FALSE(move_target.contains("old"));
    EXPECT_EQ(move_target.to_json_string(), source.to_json_string());
}

TEST(JsonContainerTest, primitive_values) {
    JsonContainer bool_json = BOOL_VALUE;
    JsonContainer int_json = INT_VALUE;
    JsonContainer int64_json = INT64_VALUE;
    JsonContainer double_json = DOUBLE_VALUE;
    JsonContainer float_json = FLOAT_VALUE;
    JsonContainer string_json = TEST_STRING;
    JsonContainer c_string_json = C_STRING_VALUE;
    JsonContainer null_json = nullptr;

    EXPECT_TRUE(bool_json.is_boolean());
    EXPECT_EQ(bool_json.type_name(), "boolean");
    EXPECT_TRUE(bool_json.as_bool().has_value());
    EXPECT_EQ(bool_json.as_bool().value(), BOOL_VALUE);
    EXPECT_EQ(bool_json.get_bool(), BOOL_VALUE);

    EXPECT_TRUE(int_json.is_number());
    EXPECT_TRUE(int_json.is_number_integer());
    EXPECT_FALSE(int_json.is_number_float());
    EXPECT_EQ(int_json.type_name(), "number");
    EXPECT_TRUE(int_json.as_int().has_value());
    EXPECT_EQ(int_json.as_int().value(), INT_VALUE);
    EXPECT_EQ(int_json.get_int(), INT_VALUE);

    EXPECT_TRUE(int64_json.is_number());
    EXPECT_TRUE(int64_json.is_number_integer());
    EXPECT_FALSE(int64_json.is_number_float());
    EXPECT_EQ(int64_json.type_name(), "number");
    EXPECT_TRUE(int64_json.as_int().has_value());
    EXPECT_EQ(int64_json.as_int().value(), INT64_VALUE);
    EXPECT_EQ(int64_json.get_int(), INT64_VALUE);

    EXPECT_TRUE(double_json.is_number());
    EXPECT_TRUE(double_json.is_number_float());
    EXPECT_FALSE(double_json.is_number_integer());
    EXPECT_EQ(double_json.type_name(), "number");
    EXPECT_TRUE(double_json.as_double().has_value());
    EXPECT_DOUBLE_EQ(double_json.as_double().value(), DOUBLE_VALUE);
    EXPECT_DOUBLE_EQ(double_json.get_double(), DOUBLE_VALUE);

    EXPECT_TRUE(float_json.is_number());
    EXPECT_TRUE(float_json.is_number_float());
    EXPECT_FALSE(float_json.is_number_integer());
    EXPECT_EQ(float_json.type_name(), "number");
    EXPECT_TRUE(float_json.as_double().has_value());
    EXPECT_FLOAT_EQ(static_cast<float>(float_json.as_double().value()), FLOAT_VALUE);
    EXPECT_FLOAT_EQ(static_cast<float>(float_json.get_double()), FLOAT_VALUE);

    EXPECT_TRUE(string_json.is_string());
    EXPECT_EQ(string_json.type_name(), "string");
    EXPECT_TRUE(string_json.as_string().has_value());
    EXPECT_EQ(string_json.as_string().value(), TEST_STRING);
    EXPECT_EQ(string_json.get_string(), TEST_STRING);

    EXPECT_TRUE(c_string_json.is_string());
    EXPECT_EQ(c_string_json.type_name(), "string");
    EXPECT_TRUE(c_string_json.as_string().has_value());
    EXPECT_EQ(c_string_json.as_string().value(), C_STRING_VALUE);
    EXPECT_EQ(c_string_json.get_string(), C_STRING_VALUE);

    EXPECT_TRUE(null_json.is_null());
    EXPECT_EQ(null_json.type_name(), "null");
    EXPECT_EQ(null_json.size(), 0);
    EXPECT_EQ(null_json.empty(), true);
    null_json = "not null";
    EXPECT_EQ(null_json.get_string(), "not null");
    null_json = nullptr;
    EXPECT_TRUE(null_json.is_null());

    EXPECT_FALSE(string_json.as_bool().has_value());
    EXPECT_FALSE(bool_json.as_int().has_value());
    EXPECT_FALSE(int_json.as_string().has_value());

    EXPECT_THROW(string_json.get_bool(), ov::Exception);
    EXPECT_THROW(bool_json.get_int(), ov::Exception);
    EXPECT_THROW(int_json.get_string(), ov::Exception);

    // Primitives are considered as not empty
    EXPECT_FALSE(string_json.empty());
    EXPECT_FALSE(bool_json.empty());
    EXPECT_FALSE(int_json.empty());
    EXPECT_EQ(string_json.size(), 1);
    EXPECT_EQ(bool_json.size(), 1);
    EXPECT_EQ(int_json.size(), 1);
    
    // Null is considered as empty
    EXPECT_TRUE(null_json.empty());
    EXPECT_EQ(null_json.size(), 0);
}

TEST(JsonContainerTest, array_operations) {
    JsonContainer jc;
    jc.push_back(BOOL_VALUE)
      .push_back(INT_VALUE)
      .push_back(INT64_VALUE)
      .push_back(DOUBLE_VALUE)
      .push_back(FLOAT_VALUE)
      .push_back(TEST_STRING)
      .push_back(C_STRING_VALUE)
      .push_back(JsonContainer({{"nested", "value"}}));
      
    EXPECT_TRUE(jc.is_array());
    EXPECT_EQ(jc.type_name(), "array");
    EXPECT_EQ(jc.size(), 8);
    EXPECT_FALSE(jc.empty());

    EXPECT_EQ(jc[size_t(0)].get_bool(), BOOL_VALUE);
    EXPECT_EQ(jc["1"].get_int(), INT_VALUE);
    EXPECT_EQ(jc[2].get_int(), INT64_VALUE);
    EXPECT_DOUBLE_EQ(jc[3].get_double(), DOUBLE_VALUE);
    EXPECT_FLOAT_EQ(static_cast<float>(jc[4].get_double()), FLOAT_VALUE);
    EXPECT_EQ(jc[5].get_string(), TEST_STRING);
    EXPECT_EQ(jc[6].get_string(), C_STRING_VALUE);
    EXPECT_EQ(jc[7]["nested"].get_string(), "value");

    // Test array append by index
    jc[8] = "appended";
    EXPECT_EQ(jc.size(), 9);
    EXPECT_EQ(jc[8].get_string(), "appended");
    
    // Contains method checks object keys, should return false for arrays
    EXPECT_FALSE(jc.contains("8"));

    // Test erase by index with array shifting
    jc.erase(7);
    EXPECT_EQ(jc.size(), 8);
    EXPECT_EQ(jc[7].get_string(), "appended");
    EXPECT_THROW(jc.erase(100), ov::Exception);
    EXPECT_THROW(jc[0].erase(0), ov::Exception); // test erase by index for primitives

    // Test out-of-bounds access
    EXPECT_THROW(jc[100].as_string(), ov::Exception);

    // Test out-of-bounds assignment expands array with nulls
    jc.to_empty_array();
    EXPECT_EQ(jc.size(), 0);
    jc[2] = "value";  // Should create nulls at indices 0-1
    EXPECT_TRUE(jc.is_array());
    EXPECT_EQ(jc.size(), 3);
    EXPECT_TRUE(jc[0].is_null());
    EXPECT_TRUE(jc[1].is_null());
    EXPECT_TRUE(jc[1].empty());
    EXPECT_EQ(jc[1].type_name(), "null");
    EXPECT_EQ(jc[2].get_string(), "value");

    // Test clear array
    EXPECT_THROW(jc[2].clear(), ov::Exception); // test clear for primitives
    EXPECT_NO_THROW(jc.clear());
    EXPECT_TRUE(jc.is_array());
    EXPECT_EQ(jc.size(), 0);
}

TEST(JsonContainerTest, object_operations) {
    JsonContainer jc;
    jc["key1"] = "value";
    jc[std::string("key2")] = true;
    
    EXPECT_TRUE(jc.is_object());
    EXPECT_EQ(jc.type_name(), "object");
    EXPECT_EQ(jc.size(), 2);
    EXPECT_FALSE(jc.empty());
    
    // Test object contains and access
    EXPECT_TRUE(jc.contains("key1"));
    EXPECT_TRUE(jc.contains("key2"));
    EXPECT_FALSE(jc.contains("nonexistent"));
    EXPECT_NO_THROW(jc["key1"].get_string());
    EXPECT_THROW(jc["nonexistent"].get_string(), ov::Exception);
    EXPECT_THROW(jc["key1"]["nested"].get_string(), ov::Exception);

    // Test erase by key in object
    EXPECT_TRUE(jc.contains("key2"));
    EXPECT_NO_THROW(jc.erase("key2"));
    EXPECT_FALSE(jc.contains("key2"));
    EXPECT_THROW(jc.erase("key2"), ov::Exception);
    EXPECT_THROW(jc.erase("nonexistent"), ov::Exception);
    EXPECT_THROW(jc["key1"].erase("something"), ov::Exception); // test erase by key for primitives

    jc["key1"] = jc["key1"];
    EXPECT_EQ(jc["key1"].get_string(), "value");

    // Test object key assignment with integer (converted to string)
    jc.to_empty_object();
    EXPECT_EQ(jc.size(), 0);
    jc[5] = "value";
    EXPECT_TRUE(jc.is_object());
    EXPECT_EQ(jc.size(), 1);
    EXPECT_EQ(jc["5"].get_string(), "value");
    EXPECT_EQ(jc[5].get_string(), "value");

    // Test nested object creation
    EXPECT_NO_THROW(jc["new_path"]["deep"]["nested"] = "value");
    EXPECT_EQ(jc["new_path"]["deep"]["nested"].get_string(), "value");

    // Test clear object
    jc["key"] = "value";
    EXPECT_THROW(jc["key"].clear(), ov::Exception); // test clear for primitives
    EXPECT_NO_THROW(jc.clear());
    EXPECT_TRUE(jc.is_object());
    EXPECT_EQ(jc.size(), 0);
}

TEST(JsonContainerTest, container_conversion) {
    JsonContainer jc = "primitive";
    EXPECT_TRUE(jc.is_string());
    EXPECT_EQ(jc.get_string(), "primitive");

    // Test primitive to array conversion
    jc.to_empty_array();
    EXPECT_TRUE(jc.is_array());
    EXPECT_EQ(jc.size(), 0);
    jc.push_back("item1");
    EXPECT_EQ(jc.size(), 1);
    EXPECT_EQ(jc[0].get_string(), "item1");

    // Test array to object conversion
    jc.to_empty_object();
    EXPECT_TRUE(jc.is_object());
    EXPECT_EQ(jc.size(), 0);
    jc["key"] = "value";
    EXPECT_EQ(jc.size(), 1);
    EXPECT_EQ(jc["key"].get_string(), "value");
}

TEST(JsonContainerTest, nested_structures) {
    JsonContainer jc;
    jc["users"][0]["name"] = "Alice";
    jc["users"][0]["age"] = 30;
    jc["users"][0]["hobbies"].push_back("reading").push_back("coding");
    
    jc["users"][1]["name"] = "Bob";
    jc["users"][1]["active"] = false;
    
    jc["metadata"]["counts"]["total"] = 2;
    
    EXPECT_TRUE(jc["users"].is_array());
    EXPECT_TRUE(jc["users"][0].is_object());
    EXPECT_EQ(jc["users"][0]["name"].get_string(), "Alice");
    EXPECT_EQ(jc["users"][0]["age"].get_int(), 30);
    EXPECT_TRUE(jc["users"][0]["hobbies"].is_array());
    EXPECT_EQ(jc["users"][0]["hobbies"].size(), 2);
    EXPECT_EQ(jc["users"][0]["hobbies"][1].get_string(), "coding");
    EXPECT_TRUE(jc["users"][1].is_object());
    EXPECT_EQ(jc["users"][1]["name"].get_string(), "Bob");
    EXPECT_EQ(jc["users"][1]["active"].get_bool(), false);
    EXPECT_TRUE(jc["metadata"].is_object());
    EXPECT_EQ(jc["metadata"]["counts"]["total"].get_int(), 2);

    JsonContainer root;
    root["nested"] = jc;
    EXPECT_EQ(root["nested"]["users"][0]["name"].get_string(), "Alice");
}

TEST(JsonContainerTest, json_string) {
    JsonContainer any_map_json({
        {"string", "test"},
        {"number", 42},
        {"boolean", true},
        {"array", JsonContainer().to_empty_array().push_back(1).push_back(2)},
        {"object", JsonContainer({{"nested", "value"}})}
    });
    
    std::string any_map_json_string = any_map_json.to_json_string();
    // AnyMap sorts keys alphabetically
    EXPECT_EQ(any_map_json_string, R"({"array":[1,2],"boolean":true,"number":42,"object":{"nested":"value"},"string":"test"})");

    JsonContainer jc;
    jc["string"] = "test";
    jc["number"] = 42;
    jc["boolean"] = true;
    jc["array"].to_empty_array().push_back(1).push_back(2);
    jc["object"]["nested"] = "value";

    std::string json_string = jc.to_json_string();
    // Keys order should be preserved
    EXPECT_EQ(json_string, R"({"string":"test","number":42,"boolean":true,"array":[1,2],"object":{"nested":"value"}})");
    
    std::string json_string_with_indent = jc.to_json_string(2);
    EXPECT_GT(json_string_with_indent.length(), json_string.length());
    
    JsonContainer parsed = JsonContainer::from_json_string(json_string);
    EXPECT_EQ(parsed, jc);
    
    EXPECT_EQ(parsed["string"].get_string(), "test");
    EXPECT_EQ(parsed["number"].get_int(), 42);
    EXPECT_EQ(parsed["boolean"].get_bool(), true);
    EXPECT_EQ(parsed["array"].size(), 2);
    EXPECT_EQ(parsed["object"]["nested"].get_string(), "value");
}

TEST(JsonContainerTest, subcontainers_modifications) {
    JsonContainer messages = JsonContainer::array();
    messages.push_back({{"role", "system"}, {"content", "assistant"}});
    messages.push_back({{"role", "user"}, {"content", "question"}});
    messages.push_back({{"role", "assistant"}, {"content", "answer"}});

    auto middle = messages[1];
    middle["content"] = "modified_question";
    EXPECT_EQ(messages[1]["content"].get_string(), "modified_question");

    messages.erase(1);
    messages.erase(1);
    EXPECT_THROW(middle["content"].get_string(), ov::Exception);
}

TEST(JsonContainerTest, json_serialization) {
    JsonContainer root = JsonContainer::object();
    root["user"]["name"] = "Alice";
    root["user"]["age"] = 30;
    root["system"]["version"] = "1.0";
    
    JsonContainer user = root["user"];
    
    nlohmann::ordered_json json = user;
    
    EXPECT_TRUE(json.contains("name"));
    EXPECT_TRUE(json.contains("age"));
    EXPECT_FALSE(json.contains("system"));

    EXPECT_EQ(json["name"].get<std::string>(), "Alice");
    EXPECT_EQ(json["age"].get<int>(), 30);
}
