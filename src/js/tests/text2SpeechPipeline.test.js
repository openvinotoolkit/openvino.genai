// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { Text2SpeechPipeline } from "../dist/index.js";
import { Text2SpeechPipeline as Text2SpeechPipelineClass } from "../dist/pipelines/text2SpeechPipeline.js";

const { TTS_MODEL_PATH } = process.env;

if (!TTS_MODEL_PATH) {
  throw new Error(
    "Environment variable TTS_MODEL_PATH must be set to the TTS model directory for tests.",
  );
}

describe("Text2SpeechPipeline methods", () => {
  let pipeline;

  before(async () => {
    pipeline = await Text2SpeechPipeline(TTS_MODEL_PATH, "CPU");
  });

  it("generate(text) returns speeches array", async () => {
    const result = await pipeline.generate("Hello OpenVINO");
    assert.ok(result);
    assert.ok(Array.isArray(result.speeches));
    assert.strictEqual(result.speeches.length, 1);
    assert.ok(
      result.speeches[0].data instanceof Float32Array,
      "speech.data should be Float32Array",
    );
    assert.ok(result.speeches[0].data.length > 0, "speech.data should not be empty");
  });

  it("generate(texts) handles batch of texts", async () => {
    const result = await pipeline.generate(["Hello", "World"]);
    assert.ok(result);
    assert.ok(Array.isArray(result.speeches));
    assert.strictEqual(result.speeches.length, 2);
    for (const speech of result.speeches) {
      assert.ok(speech.data instanceof Float32Array, "speech.data should be Float32Array");
      assert.ok(speech.data.length > 0, "speech.data should not be empty");
    }
  });

  it("generate(text, { generationConfig }) accepts generation config", async () => {
    const result = await pipeline.generate("Test speech", {
      generationConfig: { maxlenratio: 20.0 },
    });
    assert.ok(result.speeches.length > 0);
  });

  it("getGenerationConfig() returns config object", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(config && typeof config === "object");
    assert.strictEqual(typeof config.minlenratio, "number");
    assert.strictEqual(typeof config.maxlenratio, "number");
    assert.strictEqual(typeof config.threshold, "number");
  });

  it("setGenerationConfig(config) updates config", () => {
    const original = pipeline.getGenerationConfig();
    const newThreshold = 0.6;
    pipeline.setGenerationConfig({ threshold: newThreshold });
    const updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.threshold, newThreshold);
    pipeline.setGenerationConfig({ threshold: original.threshold });
  });

  it("Text2SpeechPerfMetrics has base getters and TTS-specific ones", async () => {
    const result = await pipeline.generate("Performance test");
    const pm = result.perfMetrics;
    assert.ok(pm);
    assert.strictEqual(typeof pm.getLoadTime(), "number");

    const throughput = pm.getThroughput();
    assert.ok(throughput && typeof throughput === "object");
    assert.strictEqual(typeof throughput.mean, "number");
    assert.strictEqual(typeof throughput.std, "number");

    const generateDuration = pm.getGenerateDuration();
    assert.ok(generateDuration && typeof generateDuration === "object");
    assert.strictEqual(typeof generateDuration.mean, "number");
    assert.strictEqual(typeof generateDuration.std, "number");
  });

  it("Text2SpeechPerfMetrics.getNumGeneratedSamples() returns a number", async () => {
    const result = await pipeline.generate("Sample count test");
    const numSamples = result.perfMetrics.getNumGeneratedSamples();
    assert.strictEqual(typeof numSamples, "number");
    assert.ok(numSamples >= 0);
  });
});

describe("Text2SpeechPipeline initialization", () => {
  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new Text2SpeechPipelineClass(TTS_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.generate("hello"),
      /Text2SpeechPipeline is not initialized/,
    );
  });
});
