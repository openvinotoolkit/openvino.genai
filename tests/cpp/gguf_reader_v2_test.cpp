#include <gtest/gtest.h>
#include "gguf_utils/gguf_reader_v2.hpp"

TEST(GGUFReaderV2Test, DefaultConstructionSucceeds) {
    EXPECT_NO_THROW({
        ov::genai::GGUFReaderV2 reader;
    });
}

TEST(GGUFReaderV2Test, MultipleInstancesDoNotCrash) {
    EXPECT_NO_THROW({
        ov::genai::GGUFReaderV2 reader1;
        ov::genai::GGUFReaderV2 reader2;
        ov::genai::GGUFReaderV2 reader3;
    });
}

TEST(GGUFReaderV2Test, InvalidPathThrows) {
    ov::genai::GGUFReaderV2 reader;
    EXPECT_THROW(
        reader.read("non_existent_fake_path.gguf"),
        std::runtime_error
    );
}

TEST(GGUFReaderV2Test, DestructorCleanupIsCorrect) {
    EXPECT_NO_THROW({
        {
            ov::genai::GGUFReaderV2 reader;
        }
    });
}

TEST(GGUFReaderV2Test, RealModelLoadsIfEnvVarSet) {
    const char* model_path = std::getenv("GGUF_TEST_MODEL");
    if (!model_path) {
        GTEST_SKIP() << "GGUF_TEST_MODEL not set, skipping real model test";
    }
    ov::genai::GGUFReaderV2 reader;
    EXPECT_NO_THROW(reader.read(model_path));
}