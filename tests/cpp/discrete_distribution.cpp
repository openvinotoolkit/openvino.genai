// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>

// std::mt19937 gives same output on different platforms
// std::discrete_distribution is a platfrom specific
// std::discrete_distribution used for random sampling
// so random sampling tests are platfrom specific as well

TEST(TestDiscreteDisctibution, random_gen) {
    std::mt19937 rng_engine;

    rng_engine.seed(0);

    std::vector<size_t> EXPECTED{2357136044,
                                 2546248239,
                                 3071714933,
                                 3626093760,
                                 2588848963,
                                 3684848379,
                                 2340255427,
                                 3638918503,
                                 1819583497,
                                 2678185683};

    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(rng_engine(), EXPECTED[i]);
    }
}

TEST(TestDiscreteDisctibution, discrete_distribution) {
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
