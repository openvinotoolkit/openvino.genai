// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { WhisperPipeline, StreamingStatus } from "../dist/index.js";
import { createTestRawSpeech } from "./utils.js";
import { WhisperPipeline as WhisperPipelineClass } from "../dist/pipelines/whisperPipeline.js";

const { WHISPER_MODEL_PATH } = process.env;

if (!WHISPER_MODEL_PATH) {
  throw new Error(
    "Environment variable WHISPER_MODEL_PATH must be set to the Whisper model directory for tests.",
  );
}
describe("WhisperPipeline creation", () => {
  it("WhisperPipeline(modelPath, device) creates and initializes pipeline", async () => {
    const pipeline = await WhisperPipeline(WHISPER_MODEL_PATH, "CPU");
    assert.ok(pipeline);
    assert.strictEqual(typeof pipeline.generate, "function");
    assert.strictEqual(typeof pipeline.getTokenizer, "function");
    assert.strictEqual(typeof pipeline.getGenerationConfig, "function");
    assert.strictEqual(typeof pipeline.setGenerationConfig, "function");
  });

  it("WhisperPipeline(modelPath, device, properties) accepts optional properties", async () => {
    const pipeline = await WhisperPipeline(WHISPER_MODEL_PATH, "CPU", {});
    assert.ok(pipeline);
  });
});

describe("WhisperPipeline methods", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await WhisperPipeline(WHISPER_MODEL_PATH, "CPU");
    rawSpeech = createTestRawSpeech();
  });

  it("generate(rawSpeech) returns texts, scores and perfMetrics", async () => {
    const result = await pipeline.generate(rawSpeech);
    assert.ok(Array.isArray(result.texts));
    assert.ok(Array.isArray(result.scores));
    assert.strictEqual(result.texts.length, result.scores.length);
    assert.ok(result.perfMetrics);
    assert.strictEqual(typeof result.perfMetrics.getLoadTime(), "number");
    const ttft = result.perfMetrics.getTTFT();
    assert.ok(ttft && typeof ttft === "object");
    assert.strictEqual(typeof ttft.mean, "number");
    assert.strictEqual(typeof ttft.std, "number");
    const tpot = result.perfMetrics.getTPOT();
    assert.ok(tpot && typeof tpot === "object");
    assert.strictEqual(typeof tpot.mean, "number");
    assert.strictEqual(typeof tpot.std, "number");
  });

  it("generate(rawSpeech, options) accepts generationConfig", async () => {
    const result = await pipeline.generate(rawSpeech, {
      generationConfig: { language: "<|en|>", task: "transcribe" },
    });
    assert.ok(Array.isArray(result.texts));
    assert.ok(result.perfMetrics);
  });

  it("generate(rawSpeech, options) with streamer calls streamer with chunks", async () => {
    const chunks = [];
    const result = await pipeline.generate(rawSpeech, {
      streamer: (chunk) => {
        chunks.push(chunk);
        assert.strictEqual(typeof chunk, "string");
        return StreamingStatus.RUNNING;
      },
    });
    assert.ok(chunks.length > 0);
    assert.strictEqual(chunks.join(""), result.texts[0]);
  });

  it("stream(rawSpeech) returns async iterator of chunks", async () => {
    const chunks = [];
    const stream = pipeline.stream(rawSpeech);
    for await (const chunk of stream) {
      chunks.push(chunk);
      assert.strictEqual(typeof chunk, "string");
    }
    assert.ok(Array.isArray(chunks));
    assert.ok(chunks.length > 0);
  });

  it("stream(rawSpeech, options) accepts generation config", async () => {
    const chunks = [];
    const generationConfig = { language: "<|en|>", task: "transcribe" };
    const stream = pipeline.stream(rawSpeech, { generationConfig });
    for await (const chunk of stream) {
      chunks.push(chunk);
      assert.strictEqual(typeof chunk, "string");
    }
    assert.ok(Array.isArray(chunks));
    assert.ok(chunks.length > 0);
  });

  it("getTokenizer() returns tokenizer instance", () => {
    const tokenizer = pipeline.getTokenizer();
    assert.ok(tokenizer);
    assert.strictEqual(typeof tokenizer.encode, "function");
    assert.strictEqual(typeof tokenizer.decode, "function");
  });

  it("getGenerationConfig() returns config object", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(config && typeof config === "object");
    assert.strictEqual(typeof config.return_timestamps, "boolean");
    assert.strictEqual(typeof config.max_new_tokens, "bigint");
  });

  it("setGenerationConfig(config) updates config", () => {
    const newConfig = { initial_prompt: "hello", task: "transcribe" };
    pipeline.setGenerationConfig(newConfig);
    const config = pipeline.getGenerationConfig();
    assert.strictEqual(config.initial_prompt, newConfig.initial_prompt);
    assert.strictEqual(config.task, newConfig.task);
  });

  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new WhisperPipelineClass("/nonexistent", "CPU");
    await assert.rejects(
      uninitializedPipeline.generate(new Float32Array(100)),
      /WhisperPipeline is not initialized/,
    );
  });
});

describe("WhisperPipeline with word_timestamps=true", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await WhisperPipeline(WHISPER_MODEL_PATH, "CPU", { word_timestamps: true });
    rawSpeech = createTestRawSpeech();
  });

  it("getGenerationConfig() returns word_timestamps: true", () => {
    const config = pipeline.getGenerationConfig();
    assert.strictEqual(config.word_timestamps, true);
  });

  it("generate() without generationConfig doesn't returns chunks but returns words", async () => {
    const result = await pipeline.generate(rawSpeech);
    assert.strictEqual(result.chunks, undefined);
    assert.ok(Array.isArray(result.words));
    for (const w of result.words) {
      assert.strictEqual(typeof w.word, "string");
      assert.strictEqual(typeof w.startTs, "number");
      assert.strictEqual(typeof w.endTs, "number");
      assert.ok(w.tokenIds instanceof BigInt64Array);
      assert.ok(w.tokenIds.every((id) => typeof id === "bigint"));
    }
  });

  it("generate() with return_timestamps returns chunks with timestamps", async () => {
    const result = await pipeline.generate(rawSpeech, {
      generationConfig: { return_timestamps: true },
    });
    assert.ok(result.chunks && result.chunks.length > 0);
    for (const chunk of result.chunks) {
      assert.strictEqual(typeof chunk.text, "string");
      assert.strictEqual(typeof chunk.startTs, "number");
      assert.strictEqual(typeof chunk.endTs, "number");
    }
  });
});

describe("WhisperPerfMetrics", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await WhisperPipeline(WHISPER_MODEL_PATH, "CPU");
    rawSpeech = createTestRawSpeech();
  });

  it("result.perfMetrics has base getters and optionally Whisper-specific ones", async () => {
    const result = await pipeline.generate(rawSpeech);
    const pm = result.perfMetrics;
    assert.ok(pm);
    assert.strictEqual(typeof pm.getLoadTime(), "number");
    const ttft = pm.getTTFT();
    assert.ok(ttft && typeof ttft === "object");
    assert.strictEqual(typeof ttft.mean, "number");
    assert.strictEqual(typeof ttft.std, "number");
    const wordTsDuration = pm.getWordLevelTimestampsProcessingDuration();
    assert.ok(wordTsDuration && typeof wordTsDuration === "object");
    assert.strictEqual(typeof wordTsDuration.mean, "number");
    assert.strictEqual(typeof wordTsDuration.std, "number");
    const raw = pm.whisperRawMetrics;
    assert.ok(raw);
    assert.ok(Array.isArray(raw.featuresExtractionDurations));
    assert.ok(Array.isArray(raw.wordLevelTimestampsProcessingDurations));
  });

  it("getFeaturesExtractionDuration() returns MeanStdPair when available", async () => {
    const result = await pipeline.generate(rawSpeech);
    const pair = result.perfMetrics.getFeaturesExtractionDuration();
    assert.ok(pair && typeof pair === "object");
    assert.strictEqual(typeof pair.mean, "number");
    assert.strictEqual(typeof pair.std, "number");
  });

  it("getWordLevelTimestampsProcessingDuration() returns MeanStdPair when available", async () => {
    const result = await pipeline.generate(rawSpeech);
    const pair = result.perfMetrics.getWordLevelTimestampsProcessingDuration();
    assert.ok(pair && typeof pair === "object");
    assert.strictEqual(typeof pair.mean, "number");
    assert.strictEqual(typeof pair.std, "number");
  });
});
