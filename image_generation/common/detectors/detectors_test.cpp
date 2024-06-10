#include <gtest/gtest.h>

#include "openpose_detector.hpp"

TEST(OpenposeDetectorTest, ForwardFunction) {
    OpenposeDetector detector;
    detector.load("model");
    detector.forward("scripts/im.txt", 512, 768, 3);
}
