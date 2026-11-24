// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Tests of Text2SpeechStreamingSession on the Qwen3-TTS SDPA backend. They need Qwen3-TTS model directories (the
// tiny-random exports work) and skip otherwise:
//   QWEN3_TTS_CUSTOMVOICE_DIR, QWEN3_TTS_VOICEDESIGN_DIR, QWEN3_TTS_BASE_DIR
// QWEN3_TTS_NUM_CODE_GROUPS (default 4, the tiny-random models; 16 for the real ones), and, for the tokenizer
// extension, OPENVINO_TOKENIZERS_PATH_GENAI.

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/speech_generation/text2speech_streaming_session.hpp"

namespace {

using namespace ov::genai;

const std::string TEXT = "Okay, let's run some tests! Is everything okay? Yes, it should be fine, thank you.";
const std::string SHORT_TEXT = "Hi there.";

std::unique_ptr<Text2SpeechPipeline> make_pipeline(const char* variable) {
    const char* dir = std::getenv(variable);
    if (dir == nullptr) {
        return nullptr;
    }
    return std::make_unique<Text2SpeechPipeline>(dir, "CPU", ov::AnyMap{{"ATTENTION_BACKEND", "SDPA"}});
}

#define REQUIRE_PIPELINE(pipe, variable)                          \
    auto pipe = make_pipeline(variable);                          \
    if (!(pipe)) {                                                \
        GTEST_SKIP() << variable << " is not set";                \
    }

// The stateless codec decoder re-decodes left context per chunk, so its audio depends on where the chunks fall, which
// depends on when the text arrives. Exact equality needs the stateful decoder, the default when the model has it.
#define REQUIRE_STATEFUL_DECODER()                                                         do {                                                                                       const char* decoder = std::getenv("OV_GENAI_QWEN_TTS_STREAM_DECODER");                 if (decoder != nullptr && std::string(decoder) == "stateless") {                           GTEST_SKIP() << "exact equality needs the stateful codec decoder";                 }                                                                                  } while (false)

SpeechGenerationConfig base_config() {
    SpeechGenerationConfig config;
    config.non_streaming_mode = false;  // the layout a session uses; offline has to match it
    config.do_sample = false;
    config.subtalker_dosample = false;
    config.max_new_tokens = 32;
    config.rng_seed = 42;
    return config;
}

std::vector<float> to_vector(const ov::Tensor& tensor) {
    const float* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<std::string> split_every(const std::string& text, size_t n) {
    std::vector<std::string> pieces;
    for (size_t pos = 0; pos < text.size(); pos += n) {
        pieces.push_back(text.substr(pos, n));
    }
    return pieces;
}

// Pieces that end right before a space, like the words of a stream.
std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> pieces;
    size_t start = 0;
    for (size_t pos = 1; pos <= text.size(); ++pos) {
        if (pos == text.size() || (text[pos] == ' ' && text[pos - 1] != ' ')) {
            pieces.push_back(text.substr(start, pos - start));
            start = pos;
        }
    }
    return pieces;
}

struct SessionRun {
    std::vector<float> audio;
    size_t finals = 0;
    size_t audio_results_before_end = 0;
    GenerationFinishReason finish_reason = GenerationFinishReason::NONE;
};

SessionRun run_session(Text2SpeechPipeline& pipe,
                       const ov::Tensor& speaker_embedding,
                       const SpeechGenerationConfig& config,
                       const std::vector<std::string>& pieces) {
    Text2SpeechStreamingSession session(pipe, speaker_embedding, config);
    SessionRun run;
    bool ended = false;
    auto drain = [&]() {
        while (auto result = session.read()) {
            EXPECT_EQ(result->start_sample, run.audio.size());
            EXPECT_EQ(result->output_sample_rate, 24000u);
            const auto samples = to_vector(result->speech);
            if (!ended && !samples.empty()) {
                ++run.audio_results_before_end;
            }
            run.audio.insert(run.audio.end(), samples.begin(), samples.end());
            if (result->is_final) {
                ++run.finals;
                run.finish_reason = result->finish_reason;
                break;
            } else {
                EXPECT_NE(result->finish_reason, GenerationFinishReason::STOP);
            }
        }
    };
    for (const auto& piece : pieces) {
        session.push(piece);
        drain();
    }
    session.end();
    ended = true;
    drain();
    EXPECT_FALSE(session.read().has_value());  // nothing after the final result
    return run;
}

// A session has to give the audio of the complete text, however the text is split.
void expect_session_equals_offline(Text2SpeechPipeline& pipe,
                                   const ov::Tensor& speaker_embedding,
                                   const SpeechGenerationConfig& config,
                                   const std::string& text) {
    const auto offline = to_vector(pipe.generate(text, speaker_embedding, config).speeches.at(0));
    ASSERT_FALSE(offline.empty());
    const std::vector<std::pair<std::string, std::vector<std::string>>> splits = {
        {"whole text", {text}},
        {"words", split_words(text)},
        {"7 characters", split_every(text, 7)},
        {"1 character", split_every(text, 1)},
    };
    for (const auto& [name, pieces] : splits) {
        const SessionRun run = run_session(pipe, speaker_embedding, config, pieces);
        EXPECT_EQ(run.finals, 1u) << name;
        ASSERT_EQ(run.audio.size(), offline.size()) << name << ": sample count";
        EXPECT_TRUE(run.audio == offline) << name << ": audio differs from generate()";
    }
}

ov::Tensor random_embedding(const ov::Shape& shape) {
    ov::Tensor embedding(ov::element::f32, shape);
    std::mt19937 rng(11);
    std::normal_distribution<float> distribution(0.0f, 0.5f);
    for (size_t i = 0; i < embedding.get_size(); ++i) {
        embedding.data<float>()[i] = distribution(rng);
    }
    return embedding;
}

}  // namespace

TEST(Qwen3TTSStreamingSession, CustomVoiceEqualsOfflineGeneration) {
    REQUIRE_STATEFUL_DECODER();
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_CUSTOMVOICE_DIR");
    auto config = base_config();
    config.speaker = "serena";
    expect_session_equals_offline(*pipe, ov::Tensor(), config, TEXT);
    expect_session_equals_offline(*pipe, ov::Tensor(), config, SHORT_TEXT);
}

// With sampling, a session still gives the same audio however the text is split. It is not always the audio of
// generate(): generate() lets the model end the utterance at any step, a session only after the text has ended, so
// they differ when the sampled model ends early (a random tiny model does). They agree when generate() ran as long.
TEST(Qwen3TTSStreamingSession, SampledAudioDoesNotDependOnTheSplit) {
    REQUIRE_STATEFUL_DECODER();
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_CUSTOMVOICE_DIR");
    auto config = base_config();
    config.speaker = "serena";
    config.do_sample = true;
    config.subtalker_dosample = true;
    const auto whole = run_session(*pipe, ov::Tensor(), config, {TEXT}).audio;
    ASSERT_FALSE(whole.empty());
    for (const auto& pieces : {split_words(TEXT), split_every(TEXT, 7), split_every(TEXT, 1)}) {
        EXPECT_TRUE(run_session(*pipe, ov::Tensor(), config, pieces).audio == whole);
    }
    const auto offline = to_vector(pipe->generate(TEXT, ov::Tensor(), config).speeches.at(0));
    if (offline.size() == whole.size()) {
        EXPECT_TRUE(offline == whole);
    } else {
        std::cout << "[  NOTE  ] generate() ended the sampled utterance earlier (" << offline.size() << " samples) than the session ("
                  << whole.size() << "); the session does not stop before the text has ended" << std::endl;
    }
}

TEST(Qwen3TTSStreamingSession, VoiceDesignEqualsOfflineGeneration) {
    REQUIRE_STATEFUL_DECODER();
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_VOICEDESIGN_DIR");
    auto config = base_config();
    config.instruct = "Speak with a calm, friendly, and reassuring tone.";
    expect_session_equals_offline(*pipe, ov::Tensor(), config, TEXT);
}

TEST(Qwen3TTSStreamingSession, BaseXVectorEqualsOfflineGeneration) {
    REQUIRE_STATEFUL_DECODER();
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_BASE_DIR");
    const auto embedding = random_embedding(pipe->get_speaker_embedding_shape());
    expect_session_equals_offline(*pipe, embedding, base_config(), TEXT);
}

TEST(Qwen3TTSStreamingSession, BaseVoiceCloneEqualsOfflineGeneration) {
    REQUIRE_STATEFUL_DECODER();
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_BASE_DIR");
    const auto embedding = random_embedding(pipe->get_speaker_embedding_shape());
    auto config = base_config();
    config.ref_text = "This is a dummy-generated waveform for parity testing";
    // Synthetic reference codes; the number of code groups is a property of the model (4 for the tiny-random one).
    const char* groups_env = std::getenv("QWEN3_TTS_NUM_CODE_GROUPS");
    const size_t groups = groups_env != nullptr ? static_cast<size_t>(std::atoi(groups_env)) : 4;
    config.ref_codec_ids = ov::Tensor(ov::element::i64, ov::Shape{20, groups});
    std::mt19937 rng(5);
    for (size_t i = 0; i < config.ref_codec_ids.get_size(); ++i) {
        config.ref_codec_ids.data<int64_t>()[i] = static_cast<int64_t>(rng() % 100);
    }
    // A text long enough to cover the reference when the prefill is built, and one shorter than that.
    expect_session_equals_offline(*pipe, embedding, config, TEXT);
    expect_session_equals_offline(*pipe, embedding, config, SHORT_TEXT);
}

TEST(Qwen3TTSStreamingSession, AudioArrivesBeforeTheTextEnds) {
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_CUSTOMVOICE_DIR");
    auto config = base_config();
    config.speaker = "serena";
    config.max_new_tokens = 64;
    const SessionRun run = run_session(*pipe, ov::Tensor(), config, split_words(TEXT));
    EXPECT_GT(run.audio_results_before_end, 0u);
}

TEST(Qwen3TTSStreamingSession, FinishReasonIsLengthWhenTheLimitIsReached) {
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_CUSTOMVOICE_DIR");
    auto config = base_config();
    config.speaker = "serena";
    config.max_new_tokens = 6;
    const SessionRun run = run_session(*pipe, ov::Tensor(), config, {TEXT});
    EXPECT_EQ(run.finals, 1u);
    EXPECT_EQ(run.finish_reason, GenerationFinishReason::LENGTH);
}

TEST(Qwen3TTSStreamingSession, Lifecycle) {
    REQUIRE_PIPELINE(pipe, "QWEN3_TTS_CUSTOMVOICE_DIR");
    auto config = base_config();
    config.speaker = "serena";

    {
        Text2SpeechStreamingSession session(*pipe, ov::Tensor(), config);
        EXPECT_FALSE(session.read().has_value());  // nothing pushed yet
        session.push("Hello");
        EXPECT_FALSE(session.read().has_value());  // "Hello" may still grow into a longer word
        // Only one session per pipeline, and generate() cannot run next to it.
        EXPECT_THROW(Text2SpeechStreamingSession(*pipe, ov::Tensor(), config), ov::Exception);
        EXPECT_THROW(pipe->generate("Hello", ov::Tensor(), config), ov::Exception);
        session.end();
        EXPECT_THROW(session.end(), ov::Exception);
        EXPECT_THROW(session.push("more"), ov::Exception);
        size_t finals = 0;
        while (auto result = session.read()) {
            finals += result->is_final ? 1 : 0;
        }
        EXPECT_EQ(finals, 1u);
    }
    // The slot is free again after the session is gone.
    EXPECT_NO_THROW(pipe->generate("Hello", ov::Tensor(), config));

    {
        // A session that ends without any text cannot speak.
        Text2SpeechStreamingSession empty(*pipe, ov::Tensor(), config);
        empty.end();
        EXPECT_THROW(empty.read(), ov::Exception);
    }

    {
        // Moving a session keeps it usable; the moved-from one refuses.
        Text2SpeechStreamingSession first(*pipe, ov::Tensor(), config);
        Text2SpeechStreamingSession second(std::move(first));
        EXPECT_THROW(first.push("x"), ov::Exception);
        second.push("Moved sessions work fine.");
        second.end();
        size_t finals = 0;
        while (auto result = second.read()) {
            finals += result->is_final ? 1 : 0;
        }
        EXPECT_EQ(finals, 1u);
    }
}
