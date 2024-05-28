#include <gtest/gtest.h>

#include "openpose_detector.hpp"

TEST(OpenposeDetectorTest, FooFunction) {
    OpenposeDetector detector;
    int result = detector.foo();
    EXPECT_EQ(result, 1) << "result should be 1";
}

TEST(OpenposeDetectorTest, LoadBGRFunction) {
    OpenposeDetector detector;
    detector.load_bgr("scripts/im.txt", 512, 768, 3);
}
