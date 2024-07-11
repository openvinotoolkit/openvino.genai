// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>

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

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        EXPECT_EQ(random_id, 1);
#elif __APPLE__
        EXPECT_EQ(random_id, 1);
#elif __linux__
        EXPECT_EQ(random_id, 0);
#endif

    }
}
