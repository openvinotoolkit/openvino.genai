#include <gtest/gtest.h>

#include "openpose_detector.hpp"

TEST(OpenposeDetectorTest, FooFunction) {
    auto detector = OpenposeDetector();
    int result = detector.foo();
    EXPECT_EQ(result, 1) << "result should be 1";
}
