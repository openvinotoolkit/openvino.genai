// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/model_identity.hpp"
#include "continuous_batching/cache/model_identity_memo.hpp"
#include "helper.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov::genai;

namespace {

/// A value type with an attribute adapter that the model identity has no way to hash.
struct OpaqueValue {
    int value = 0;
};

}  // namespace

namespace ov {
template <>
class AttributeAdapter<OpaqueValue> : public DirectValueAccessor<OpaqueValue> {
public:
    explicit AttributeAdapter(OpaqueValue& value) : DirectValueAccessor<OpaqueValue>(value) {}
    OPENVINO_RTTI("AttributeAdapter<OpaqueValue>");
};
}  // namespace ov

namespace {

class OpaqueAttributeOp : public ov::op::Op {
public:
    OPENVINO_OP("OpaqueAttributeOp", "genai_test");
    OpaqueAttributeOp() = default;
    explicit OpaqueAttributeOp(const ov::Output<ov::Node>& input) : ov::op::Op({input}) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<OpaqueAttributeOp>(new_args.at(0));
    }
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("opaque", m_opaque);
        return true;
    }

private:
    OpaqueValue m_opaque;
};

/// A Loop that adds (or subtracts) a constant to its carried value three times; the outer graph is the same either way.
std::shared_ptr<ov::Model> make_loop_model(bool subtract, float body_constant = 1.f) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    auto execution_condition = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {true});

    auto body_parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto step = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, std::vector<float>(4, body_constant));
    std::shared_ptr<ov::Node> body_value;
    if (subtract) {
        body_value = std::make_shared<ov::op::v1::Subtract>(body_parameter, step);
    } else {
        body_value = std::make_shared<ov::op::v1::Add>(body_parameter, step);
    }
    auto body_result = std::make_shared<ov::op::v0::Result>(body_value);
    auto condition_result = std::make_shared<ov::op::v0::Result>(
        ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {true}));
    auto body = std::make_shared<ov::Model>(ov::ResultVector{body_result, condition_result},
                                            ov::ParameterVector{body_parameter});

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, execution_condition);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 1});
    loop->set_merged_input(body_parameter, input, body_result);
    const auto output = loop->get_iter_value(body_result, -1);
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> make_model(const std::vector<float>& weights, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name("input");
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, weights);
    constant->set_friendly_name("weight");
    auto add = std::make_shared<ov::op::v1::Add>(parameter, constant);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->set_friendly_name("output");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    model->set_friendly_name("test_model");
    return model;
}

/// Same parameter/constant pair as make_model(), but Subtract instead of Add: identical constants, different graph.
std::shared_ptr<ov::Model> make_model_with_subtract(const std::vector<float>& weights, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name("input");
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, weights);
    constant->set_friendly_name("weight");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(parameter, constant);
    subtract->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(subtract);
    result->set_friendly_name("output");
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    model->set_friendly_name("test_model");
    return model;
}

}  // namespace

TEST(TestModelIdentity, SameWeightsGiveTheSameDigest) {
    const auto first = compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2}));
    const auto second = compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2}));

    EXPECT_EQ(first.size(), 32u);
    EXPECT_EQ(first, second);
}

TEST(TestModelIdentity, DifferentWeightsGiveADifferentDigest) {
    EXPECT_NE(compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2})),
              compute_model_identity(make_model({1.f, 2.f, 3.f, 5.f}, {2, 2})));
}

TEST(TestModelIdentity, SameBytesInADifferentShapeGiveADifferentDigest) {
    EXPECT_NE(compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2})),
              compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {1, 4})));
}

TEST(TestModelIdentity, SameConstantsWithADifferentOperatorGiveADifferentDigest) {
    EXPECT_NE(compute_model_identity(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2})),
              compute_model_identity(make_model_with_subtract({1.f, 2.f, 3.f, 4.f}, {2, 2})));
}

TEST(TestModelIdentity, StringConstantsAreHashedByTheirValues) {
    const auto make_string_model = [](const std::vector<std::string>& values) {
        auto constant = ov::op::v0::Constant::create(ov::element::string, ov::Shape{values.size()}, values);
        auto result = std::make_shared<ov::op::v0::Result>(constant);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
    };

    EXPECT_EQ(compute_model_identity(make_string_model({"ab", "c"})), compute_model_identity(make_string_model({"ab", "c"})));
    EXPECT_NE(compute_model_identity(make_string_model({"ab", "c"})), compute_model_identity(make_string_model({"ab", "d"})));
    // The split between the strings counts, not only the characters.
    EXPECT_NE(compute_model_identity(make_string_model({"ab", "c"})), compute_model_identity(make_string_model({"a", "bc"})));
}

TEST(TestModelIdentity, SameLoopGivesTheSameDigest) {
    EXPECT_EQ(compute_model_identity(make_loop_model(/*subtract=*/false)),
              compute_model_identity(make_loop_model(/*subtract=*/false)));
}

TEST(TestModelIdentity, LoopsWithTheSameOuterGraphButDifferentBodiesGiveDifferentDigests) {
    EXPECT_NE(compute_model_identity(make_loop_model(/*subtract=*/false)),
              compute_model_identity(make_loop_model(/*subtract=*/true)));
}

TEST(TestModelIdentity, LoopsWithADifferentConstantInTheBodyGiveDifferentDigests) {
    EXPECT_NE(compute_model_identity(make_loop_model(/*subtract=*/false, /*body_constant=*/1.f)),
              compute_model_identity(make_loop_model(/*subtract=*/false, /*body_constant=*/2.f)));
}

TEST(TestModelIdentity, RejectsAnAttributeItCannotHash) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto opaque = std::make_shared<OpaqueAttributeOp>(parameter);
    auto result = std::make_shared<ov::op::v0::Result>(opaque);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});

    EXPECT_THROW(compute_model_identity(model), ov::Exception);
}

// 6M floats are 24 MiB, which the digest splits into two shards that are hashed on different threads.
TEST(TestModelIdentity, ALargeConstantIsCoveredInEveryShard) {
    const ov::Shape shape{6 * 1024 * 1024};
    std::vector<float> weights(shape[0], 1.f);
    const auto base = compute_model_identity(make_model(weights, shape));
    EXPECT_EQ(base, compute_model_identity(make_model(weights, shape)));

    weights.front() = 2.f;
    EXPECT_NE(base, compute_model_identity(make_model(weights, shape)));
    weights.front() = 1.f;
    weights.back() = 2.f;
    EXPECT_NE(base, compute_model_identity(make_model(weights, shape)));
}

// The structure digest is what is known before the model is compiled; the bytes of a large constant are the weights digest's.
TEST(TestModelIdentity, StructureDigestSkipsTheMiddleOfALargeConstantAndTheWeightsDigestDoesNot) {
    const ov::Shape shape{6 * 1024 * 1024};
    std::vector<float> weights(shape[0], 1.f);
    const std::shared_ptr<const ov::Model> base = make_model(weights, shape);
    weights[shape[0] / 2] = 2.f;
    const std::shared_ptr<const ov::Model> changed = make_model(weights, shape);

    EXPECT_EQ(compute_structure_digest(base), compute_structure_digest(changed));
    EXPECT_NE(compute_weights_digest(base), compute_weights_digest(changed));
    EXPECT_NE(compute_model_identity(base), compute_model_identity(changed));
}

TEST(TestModelIdentity, StructureDigestCoversASmallConstantAndTheGraph) {
    const std::shared_ptr<const ov::Model> base = make_model({1.f, 2.f, 3.f, 4.f}, {2, 2});
    EXPECT_NE(compute_structure_digest(base), compute_structure_digest(make_model({1.f, 2.f, 3.f, 5.f}, {2, 2})));
    EXPECT_NE(compute_structure_digest(base), compute_structure_digest(make_model_with_subtract({1.f, 2.f, 3.f, 4.f}, {2, 2})));
}

TEST(TestModelIdentity, ATiedConstantIsHashedLikeTwoConstantsOfTheSameBytes) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto shared = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, std::vector<float>{1.f, 2.f, 3.f, 4.f});
    auto first = std::make_shared<ov::op::v1::Add>(parameter, shared);
    auto second = std::make_shared<ov::op::v1::Add>(first, shared);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(second)},
                                             ov::ParameterVector{parameter});

    EXPECT_EQ(compute_model_identity(model), compute_model_identity(model->clone()));
}

namespace {

/// A temporary directory with the model of make_model() saved in it as openvino_model.xml / .bin.
struct SavedModel {
    std::filesystem::path directory;
    std::filesystem::path memo_directory;
    std::shared_ptr<ov::Model> model;

    explicit SavedModel(const std::vector<float>& weights) {
        directory = std::filesystem::temp_directory_path() / ("genai_memo_test_" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(directory);
        memo_directory = directory / "memo";
        save(weights);
    }
    ~SavedModel() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    void save(const std::vector<float>& weights) {
        model = make_model(weights, {1, weights.size()});
        ov::serialize(model, (directory / "openvino_model.xml").string(), (directory / "openvino_model.bin").string());
    }
};

}  // namespace

TEST(TestModelIdentityMemo, StoresTheWeightsDigestAndUsesItNextTime) {
    SavedModel saved({1.f, 2.f, 3.f, 4.f});
    bool hit = true;
    const std::string first = compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit);
    EXPECT_FALSE(hit);
    EXPECT_EQ(first, compute_weights_digest(saved.model));

    const std::string second = compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit);
    EXPECT_TRUE(hit);
    EXPECT_EQ(second, first);
}

TEST(TestModelIdentityMemo, ChangedWeightsOfTheSameSizeComputeTheDigestAgain) {
    SavedModel saved({1.f, 2.f, 3.f, 4.f});
    bool hit = false;
    const std::string before = compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit);

    saved.save({1.f, 2.f, 3.f, 5.f});
    const std::string after = compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit);
    EXPECT_FALSE(hit);
    EXPECT_NE(after, before);
    EXPECT_EQ(after, compute_weights_digest(saved.model));
}

TEST(TestModelIdentityMemo, ADamagedMemoIsReplaced) {
    SavedModel saved({1.f, 2.f, 3.f, 4.f});
    bool hit = false;
    const std::string digest = compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit);

    const auto memo_path = weights_digest_memo_path(saved.memo_directory, saved.directory / "openvino_model.xml");
    ASSERT_TRUE(std::filesystem::exists(memo_path));
    {
        std::ofstream memo(memo_path, std::ios::trunc);
        memo << "garbage\n";
    }
    EXPECT_EQ(compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit), digest);
    EXPECT_FALSE(hit);
    EXPECT_EQ(compute_weights_digest_memoized(saved.model, saved.directory, saved.memo_directory, hit), digest);
    EXPECT_TRUE(hit);
}

TEST(TestModelIdentityMemo, AModelWithoutFilesInTheDirectoryIsNotMemoized) {
    const auto empty = std::filesystem::temp_directory_path() / ("genai_memo_empty_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(empty);
    const auto model = make_model({1.f, 2.f, 3.f, 4.f}, {2, 2});
    bool hit = true;
    EXPECT_EQ(compute_weights_digest_memoized(model, empty, empty / "memo", hit), compute_weights_digest(model));
    EXPECT_FALSE(hit);
    EXPECT_FALSE(std::filesystem::exists(empty / "memo"));
    std::filesystem::remove_all(empty);
}

TEST(TestModelIdentity, NumericsFingerprintNamesTheRuntimeAndWhatTheDeviceReports) {
    ov::Core core;
    const auto compiled = core.compile_model(make_model({1.f, 2.f, 3.f, 4.f}, {2, 2}), "CPU");
    const std::string fingerprint = compute_numerics_fingerprint(compiled, core);
    EXPECT_NE(fingerprint.find(ov::get_openvino_version().buildNumber), std::string::npos);
    EXPECT_EQ(fingerprint, compute_numerics_fingerprint(compiled, core));
}

namespace {

/// Owns what an orchestrator keeps referring to.
struct OrchestratorFixture {
    ov::Core core;
    ov::InferRequest request;
    std::shared_ptr<CacheOrchestrator> orchestrator;

    explicit OrchestratorFixture(const std::string& model_identity) {
        request = core.compile_model(get_dummy_model(core, /*num_layers=*/2)).create_infer_request();
        SchedulerConfig config;
        config.num_kv_blocks = 4;
        orchestrator = CacheOrchestrator::create(request,
                                                 config,
                                                 [](const std::string&, size_t) {
                                                     return std::numeric_limits<size_t>::max();
                                                 },
                                                 /*hidden_state_bytes_per_token=*/0,
                                                 model_identity);
    }

    KVCacheDiskLayout kv_layout() {
        return static_cast<const KVCacheManager&>(orchestrator->get_cache_manager(CacheType::KV_CACHE)).get_block_layout();
    }
};

}  // namespace

TEST(TestOffloadGroupName, IsThePlainCacheNameWithoutAModelIdentity) {
    OrchestratorFixture fixture("");

    EXPECT_EQ(fixture.orchestrator->get_offload_coordinator().group_name("kv", fixture.kv_layout(), 32, "CPU"), "kv");
}

TEST(TestOffloadGroupName, KeepsModelsDevicesAndLayoutsApart) {
    OrchestratorFixture first("model-a");
    const auto layout = first.kv_layout();
    const auto& coordinator = first.orchestrator->get_offload_coordinator();
    const auto name = coordinator.group_name("kv", layout, 32, "CPU");

    EXPECT_EQ(name.rfind("kv:", 0), 0u);
    EXPECT_EQ(name.size(), 3u + 32u);
    EXPECT_EQ(name, coordinator.group_name("kv", layout, 32, "CPU"));
    EXPECT_NE(name, coordinator.group_name("kv", layout, 16, "CPU"));
    EXPECT_NE(name, coordinator.group_name("kv", layout, 32, "GPU.0"));
    EXPECT_NE(name, coordinator.group_name("la", layout, 32, "CPU"));

    OrchestratorFixture second("model-b");
    EXPECT_NE(name, second.orchestrator->get_offload_coordinator().group_name("kv", layout, 32, "CPU"));
}

TEST(TestOffloadGroupName, KeepsSegmentLayoutsAndTypesApartEvenWithTheSameAggregateByteSize) {
    OrchestratorFixture fixture("model-a");
    const auto layout = fixture.kv_layout();
    const auto& coordinator = fixture.orchestrator->get_offload_coordinator();
    const auto name = coordinator.group_name("kv", layout, 32, "CPU", "f16 {2,2}|f16 {2,2};");

    // Two layouts can reduce to the same slot size (e.g. FP16 vs BF16, same byte width) yet mean different bytes.
    EXPECT_NE(name, coordinator.group_name("kv", layout, 32, "CPU", "bf16 {2,2}|bf16 {2,2};"));
    EXPECT_NE(name, coordinator.group_name("kv", layout, 32, "CPU", ""));
    EXPECT_EQ(name, coordinator.group_name("kv", layout, 32, "CPU", "f16 {2,2}|f16 {2,2};"));
}
