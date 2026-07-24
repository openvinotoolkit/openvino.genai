// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { ASRPipeline, StreamingStatus } from "../dist/index.js";
import { createTestRawSpeech } from "./utils.js";
import { ASRPipeline as ASRPipelineClass } from "../dist/pipelines/asrPipeline.js";

// ASRPipeline dispatches to a model implementation (e.g. Whisper or Qwen3-ASR) based on the
// model's config.json. Whisper-specific capabilities (segment/word timestamps, `<|...|>` language
// tokens, task) are validated against a Whisper model, while a dedicated block validates Qwen3-ASR
// dispatch and its `context` option.
const { ASR_MODEL_PATH, WHISPER_MODEL_PATH } = process.env;

if (!ASR_MODEL_PATH || !WHISPER_MODEL_PATH) {
  throw new Error(
    "Environment variables ASR_MODEL_PATH and WHISPER_MODEL_PATH must be set to the ASR model directories for tests.",
  );
}

describe("ASRPipeline creation (Whisper backend)", () => {
  it("ASRPipeline(modelPath, device) creates and initializes pipeline", async () => {
    const pipeline = await ASRPipeline(WHISPER_MODEL_PATH, "CPU");
    assert.ok(pipeline);
    assert.strictEqual(typeof pipeline.generate, "function");
    assert.strictEqual(typeof pipeline.getTokenizer, "function");
    assert.strictEqual(typeof pipeline.getGenerationConfig, "function");
    assert.strictEqual(typeof pipeline.setGenerationConfig, "function");
  });

  it("ASRPipeline(modelPath, device, properties) accepts optional properties", async () => {
    const pipeline = await ASRPipeline(WHISPER_MODEL_PATH, "CPU", {});
    assert.ok(pipeline);
  });
});

describe("ASRPipeline methods (Whisper backend)", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await ASRPipeline(WHISPER_MODEL_PATH, "CPU");
    rawSpeech = createTestRawSpeech();
  });

  it("generate(rawSpeech) returns texts, scores, languages and perfMetrics", async () => {
    const result = await pipeline.generate(rawSpeech);
    assert.ok(Array.isArray(result.texts));
    assert.ok(Array.isArray(result.scores));
    assert.strictEqual(result.texts.length, result.scores.length);
    assert.ok(Array.isArray(result.languages));
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

  it("setGenerationConfig(config) accepts empty suppress token arrays", () => {
    assert.doesNotThrow(() => {
      pipeline.setGenerationConfig({ suppress_tokens: [], begin_suppress_tokens: [] });
    });
  });

  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new ASRPipelineClass("/nonexistent", "CPU");
    await assert.rejects(
      uninitializedPipeline.generate(new Float32Array(100)),
      /ASRPipeline is not initialized/,
    );
  });
});

describe("ASRPipeline with word_timestamps=true (Whisper backend)", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    // For Whisper-based models the word-level timestamps capability must be enabled
    // via the constructor property; the request itself is made per generate call through
    // the generation config (word_timestamps: true).
    pipeline = await ASRPipeline(WHISPER_MODEL_PATH, "CPU", { word_timestamps: true });
    rawSpeech = createTestRawSpeech();
  });

  it("setGenerationConfig({ word_timestamps: true }) is reflected by getGenerationConfig()", () => {
    pipeline.setGenerationConfig({ word_timestamps: true });
    const config = pipeline.getGenerationConfig();
    assert.strictEqual(config.word_timestamps, true);
  });

  it("generate() with word_timestamps returns words with timestamps", async () => {
    const result = await pipeline.generate(rawSpeech, {
      generationConfig: { word_timestamps: true, language: "<|en|>", task: "transcribe" },
    });
    assert.ok(Array.isArray(result.words));
    // words are nested per input: words[inputIndex][wordIndex]
    for (const perInput of result.words) {
      assert.ok(Array.isArray(perInput));
      for (const w of perInput) {
        assert.strictEqual(typeof w.text, "string");
        assert.strictEqual(typeof w.startTs, "number");
        assert.strictEqual(typeof w.endTs, "number");
        assert.ok(w.tokenIds instanceof BigInt64Array);
        assert.ok(w.tokenIds.every((id) => typeof id === "bigint"));
      }
    }
  });

  it("generate() with return_timestamps returns chunks with timestamps", async () => {
    const result = await pipeline.generate(rawSpeech, {
      generationConfig: { return_timestamps: true },
    });
    assert.ok(result.chunks && result.chunks.length > 0);
    for (const perInput of result.chunks) {
      assert.ok(Array.isArray(perInput));
      for (const chunk of perInput) {
        assert.strictEqual(typeof chunk.text, "string");
        assert.strictEqual(typeof chunk.startTs, "number");
        assert.strictEqual(typeof chunk.endTs, "number");
      }
    }
  });
});

describe("ASRPerfMetrics (Whisper backend)", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await ASRPipeline(WHISPER_MODEL_PATH, "CPU");
    rawSpeech = createTestRawSpeech();
  });

  it("result.perfMetrics has base getters and ASR-specific ones", async () => {
    const result = await pipeline.generate(rawSpeech);
    const pm = result.perfMetrics;
    assert.ok(pm);
    assert.strictEqual(typeof pm.getLoadTime(), "number");
    const ttft = pm.getTTFT();
    assert.ok(ttft && typeof ttft === "object");
    assert.strictEqual(typeof ttft.mean, "number");
    assert.strictEqual(typeof ttft.std, "number");
    const encodeDuration = pm.getEncodeInferenceDuration();
    assert.ok(encodeDuration && typeof encodeDuration === "object");
    assert.strictEqual(typeof encodeDuration.mean, "number");
    assert.strictEqual(typeof encodeDuration.std, "number");
    const decodeDuration = pm.getDecodeInferenceDuration();
    assert.ok(decodeDuration && typeof decodeDuration === "object");
    assert.strictEqual(typeof decodeDuration.mean, "number");
    assert.strictEqual(typeof decodeDuration.std, "number");
    const raw = pm.asrRawMetrics;
    assert.ok(raw);
    assert.ok(Array.isArray(raw.featuresExtractionDurations));
    assert.ok(Array.isArray(raw.wordLevelTimestampsProcessingDurations));
    assert.ok(Array.isArray(raw.encodeInferenceDurations));
    assert.ok(Array.isArray(raw.decodeInferenceDurations));
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

describe("ASRPipeline Qwen3-ASR dispatch", () => {
  let pipeline;
  let rawSpeech;

  before(async () => {
    pipeline = await ASRPipeline(ASR_MODEL_PATH, "CPU");
    rawSpeech = createTestRawSpeech();
  });

  it("generate(rawSpeech) returns texts, scores, languages and perfMetrics", async () => {
    const result = await pipeline.generate(rawSpeech);
    assert.ok(Array.isArray(result.texts));
    assert.ok(Array.isArray(result.scores));
    assert.strictEqual(result.texts.length, result.scores.length);
    assert.ok(Array.isArray(result.languages));
    assert.ok(result.perfMetrics);
    assert.strictEqual(typeof result.perfMetrics.getLoadTime(), "number");
  });

  it("generate(rawSpeech, options) accepts the Qwen3-ASR context option", async () => {
    const result = await pipeline.generate(rawSpeech, {
      generationConfig: { context: "meeting transcript" },
    });
    assert.ok(Array.isArray(result.texts));
    assert.ok(result.perfMetrics);
  });

  it("getGenerationConfig() returns config object", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(config && typeof config === "object");
    assert.strictEqual(typeof config.return_timestamps, "boolean");
    assert.strictEqual(typeof config.max_new_tokens, "bigint");
  });
});
