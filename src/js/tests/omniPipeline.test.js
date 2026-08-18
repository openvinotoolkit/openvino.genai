// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert";
import { describe, it, before } from "node:test";
import { addon as ov } from "openvino-node";
import { createTestImageTensor, createTestRawSpeech } from "./utils.js";
import {
  OmniPipeline,
  DecodedResults,
  VLMDecodedResults,
  OmniDecodedResults,
  ChatHistory,
  StreamingStatus,
} from "../dist/index.js";

const { OMNI_PATH } = process.env;
const deterministicSpeechConfig = {
  return_audio: true,
  max_new_tokens: 4,
  talker_top_k: 1,
  cp_top_k: 1,
};

// Smoke tests: exercise the binding wiring without loading a multi-GB Qwen3-Omni
// checkpoint. Generation that requires a loaded pipeline lives in the model-gated
// suite below.
describe("OmniPipeline smoke", { skip: process.platform === "darwin" }, () => {
  it("exposes the OmniPipeline factory", () => {
    assert.strictEqual(typeof OmniPipeline, "function");
  });

  it("exposes the OmniDecodedResults class", () => {
    assert.strictEqual(typeof OmniDecodedResults, "function");
  });

  it("rejects when the model path does not exist", async () => {
    await assert.rejects(OmniPipeline("./nonexistent-omni-model", "CPU"));
  });
});

// Functional tests require a Qwen3-Omni model exported with audio output. Speech output
// uses the continuous-batching backend, which OmniPipeline enables by default. When
// OMNI_PATH is not provided the suite is skipped so the shared JS test run is not blocked
// by the large checkpoint.
describe("OmniPipeline", { skip: !OMNI_PATH || process.platform === "darwin" }, () => {
  let pipeline, testImage, testAudio;

  before(async () => {
    pipeline = await OmniPipeline(OMNI_PATH, "CPU");
    testImage = createTestImageTensor();
    const speech = createTestRawSpeech({ durationSeconds: 0.2 });
    testAudio = new ov.Tensor("f32", [speech.length], speech);
  });

  it("should generate text without speech", async () => {
    const result = await pipeline.generate("What is 2+2?", {
      textConfig: { max_new_tokens: 20 },
      talkerSpeechConfig: { return_audio: false },
    });

    assert.ok(result instanceof DecodedResults, "Result should be instance of DecodedResults");
    assert.ok(
      result instanceof VLMDecodedResults,
      "Result should be instance of VLMDecodedResults",
    );
    assert.ok(
      result instanceof OmniDecodedResults,
      "Result should be instance of OmniDecodedResults",
    );
    assert.ok(result.texts.length > 0, "Should generate some output");
    assert.strictEqual(result.speechResult.waveforms.length, 0, "No speech expected");
  });

  it("should generate text with an image and audio", async () => {
    const result = await pipeline.generate("Describe the inputs.", {
      images: [testImage],
      audios: [testAudio],
      textConfig: { max_new_tokens: 20 },
      talkerSpeechConfig: { return_audio: false },
    });
    assert.strictEqual(result.texts.length, 1);
  });

  it("should generate speech waveforms", async () => {
    const result = await pipeline.generate("Say hello.", {
      textConfig: { max_new_tokens: 20 },
      talkerSpeechConfig: deterministicSpeechConfig,
    });
    assert.ok(result.speechResult.waveforms.length > 0, "Should produce speech waveforms");
    assert.strictEqual(
      typeof result.speechResult.perfMetrics.numGeneratedSamples,
      "number",
      "Speech perf metrics should report the number of generated samples",
    );
  });

  it("should stream text and speech chunks", async () => {
    let textChunks = 0;
    let audioChunks = 0;
    const result = await pipeline.generate("Count to three.", {
      textConfig: {
        max_new_tokens: 4,
        ignore_eos: true,
        structured_output_config: { regex: "a+" },
      },
      talkerSpeechConfig: deterministicSpeechConfig,
      streamer: () => {
        textChunks++;
        return StreamingStatus.RUNNING;
      },
      speechStreamer: () => {
        audioChunks++;
        return StreamingStatus.RUNNING;
      },
    });
    assert.ok(textChunks > 0, "Text streamer should be called");
    assert.ok(audioChunks > 0, "Speech streamer should be called");
    assert.ok(result instanceof OmniDecodedResults);
  });

  it("should generate with ChatHistory", async () => {
    const history = new ChatHistory();
    history.push({ role: "user", content: "Hello" });
    const result = await pipeline.generate(history, {
      textConfig: { max_new_tokens: 20 },
      talkerSpeechConfig: { return_audio: false },
    });
    assert.ok(result instanceof OmniDecodedResults);
    assert.strictEqual(result.texts.length, 1);
  });
});
