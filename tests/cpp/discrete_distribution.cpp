// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>

TEST(TestDiscreteDisctibution, rng_engine) {
    std::mt19937 rng_engine;

    const std::vector<size_t> EXPECTED = {3499211612,
                                          581869302,
                                          3890346734,
                                          3586334585,
                                          545404204,
                                          4161255391,
                                          3922919429,
                                          949333985,
                                          2715962298,
                                          1323567403};

    for (size_t i = 0; i < 10; i++) {
        auto random_n = rng_engine();
        EXPECT_EQ(random_n, EXPECTED[i]);
    }
}

TEST(TestDiscreteDisctibution, discrete_distribution_site_t) {
    std::mt19937 rng_engine;

    std::discrete_distribution<size_t> distribution({25, 25, 25, 25});

    const std::vector<size_t> EXPECTED = {
        0, 3, 3, 0, 1, 2, 0, 3, 3, 3, 2, 3, 0, 3, 1, 0, 0, 2, 3, 2, 3, 1, 0, 2, 1, 2, 1, 1, 0, 1, 3, 1, 3, 0, 3, 3,
        0, 3, 1, 2, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 0, 2, 2, 1, 3, 2, 3, 3, 0, 3, 3, 3, 1, 1, 2, 0, 3, 0, 3, 3, 1,
        1, 0, 0, 3, 3, 3, 3, 3, 0, 3, 2, 2, 2, 2, 1, 2, 0, 2, 1, 1, 2, 1, 3, 0, 1, 1, 2, 1, 3, 1, 3, 2, 0, 1, 1, 2,
        2, 3, 2, 3, 2, 2, 2, 1, 2, 1, 3, 2, 1, 3, 0, 3, 1, 2, 1, 3, 0, 2, 2, 0, 3, 0, 1, 3, 0, 2, 0, 2, 2, 1, 0, 0,
        1, 3, 0, 1, 2, 3, 3, 3, 1, 0, 0, 2, 3, 1, 0, 0, 3, 0, 1, 1, 0, 1, 3, 2, 3, 1, 2, 0, 1, 1, 3, 3, 1, 0, 1, 1,
        2, 1, 2, 3, 1, 1, 1, 3, 2, 1, 1, 1, 3, 1, 3, 0, 1, 3, 1, 3, 2, 0, 0, 3, 3, 0, 0, 1, 3, 1, 2, 0, 3, 0, 3, 1,
        0, 0, 3, 1, 0, 2, 2, 2, 2, 1, 2, 2, 0, 0, 1, 0, 3, 0, 3, 1, 2, 1, 3, 0, 3, 0, 3, 1, 0, 0, 1, 1, 3, 0, 2, 0,
        0, 1, 1, 0, 1, 2, 0, 3, 3, 3, 0, 2, 3, 1, 0, 1, 3, 0, 2, 0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2, 1, 1, 1, 3, 1, 2,
        1, 2, 0, 1, 2, 2, 1, 3, 0, 2, 1, 1, 3, 2, 3, 3, 2, 1, 0, 0, 2, 0, 1, 3, 0, 2, 1, 1, 0, 2, 3, 0, 2, 2, 0, 1,
        1, 3, 3, 2, 0, 2, 3, 1, 3, 1, 0, 1, 0, 3, 0, 2, 3, 1, 3, 0, 3, 2, 2, 0, 2, 2, 1, 0, 2, 0, 3, 0, 1, 3, 2, 2,
        0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 0, 0, 2, 2, 3, 2, 0, 1, 0, 2, 2, 0, 1, 3, 2, 3, 3, 0, 3, 1, 3, 1, 1, 2, 2, 0,
        3, 1, 1, 1, 1, 3, 3, 2, 2, 0, 2, 2, 1, 1, 2, 3, 2, 2, 0, 1, 2, 2, 2, 0, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 3, 3,
        1, 2, 1, 1, 0, 2, 1, 3, 3, 3, 1, 3, 0, 1, 2, 0, 2, 3, 2, 1, 1, 0, 1, 3, 1, 0, 0, 3, 1, 1, 0, 2, 2, 1, 2, 3,
        1, 3, 3, 0, 3, 2, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0, 2, 2, 1, 0, 0, 3, 2, 2, 2, 0, 0, 1, 0, 0, 1, 1,
    };

    for (size_t i = 0; i < 500; i++) {
        size_t random_id = distribution(rng_engine);
        EXPECT_EQ(random_id, EXPECTED[i]);
    }
}

TEST(TestDiscreteDisctibution, discrete_distribution_float) {
    std::mt19937 rng_engine;

    rng_engine.seed(0);

    {
        std::discrete_distribution<size_t> distribution({1});

        size_t random_id = distribution(rng_engine);
        EXPECT_EQ(random_id, 0);
    }

    {
        std::discrete_distribution<size_t> distribution({0.790013, 0.209987});

        size_t random_id = distribution(rng_engine);
        EXPECT_EQ(random_id, 0);
    }
}
