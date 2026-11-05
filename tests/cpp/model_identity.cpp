// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/model_identity.hpp"
#include "helper.hpp"
#include "openvino/op/add.hpp"
#include "openvino/openvino.hpp"

using namespace ov::genai;

namespace {

std::shared_ptr<ov::Model> make_model(const std::vector<float>& weights, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto constant = ov::op::v0::Constant::create(ov::element::f32, shape, weights);
    auto add = std::make_shared<ov::op::v1::Add>(parameter, constant);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
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
                                                 model_identity);
    }

    KVCacheDiskLayout kv_layout() {
        return static_cast<KVCacheManager&>(orchestrator->get_cache_manager(CacheType::KV_CACHE)).get_block_layout();
    }
};

}  // namespace

TEST(TestOffloadGroupName, IsThePlainCacheNameWithoutAModelIdentity) {
    OrchestratorFixture fixture("");

    EXPECT_EQ(fixture.orchestrator->offload_group_name("kv", fixture.kv_layout(), 32, "CPU"), "kv");
}

TEST(TestOffloadGroupName, KeepsModelsDevicesAndLayoutsApart) {
    OrchestratorFixture first("model-a");
    const auto layout = first.kv_layout();
    const auto name = first.orchestrator->offload_group_name("kv", layout, 32, "CPU");

    EXPECT_EQ(name.rfind("kv:", 0), 0u);
    EXPECT_EQ(name.size(), 3u + 32u);
    EXPECT_EQ(name, first.orchestrator->offload_group_name("kv", layout, 32, "CPU"));
    EXPECT_NE(name, first.orchestrator->offload_group_name("kv", layout, 16, "CPU"));
    EXPECT_NE(name, first.orchestrator->offload_group_name("kv", layout, 32, "GPU.0"));
    EXPECT_NE(name, first.orchestrator->offload_group_name("la", layout, 32, "CPU"));

    OrchestratorFixture second("model-b");
    EXPECT_NE(name, second.orchestrator->offload_group_name("kv", layout, 32, "CPU"));
}
