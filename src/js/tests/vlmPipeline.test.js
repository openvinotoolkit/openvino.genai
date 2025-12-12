// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Tokenizer, VLMPipeline, DecodedResults, VLMDecodedResults } from "../dist/index.js";

import assert from "node:assert";
import { describe, it, before } from "node:test";
import { promises as fs } from "node:fs";
import { models } from "./models.js";
import { createTestImageTensor, createTestVideoTensor } from "./utils.js";

const MODEL_PATH = process.env.VLM_MODEL_PATH || `./tests/models/${models.VLLM.split("/")[1]}`;

describe("VLMPipeline", () => {
  let pipeline = null;

  before(async () => {
    try {
      await fs.access(MODEL_PATH);
    } catch {
      console.log(`Model not found at ${MODEL_PATH}, skipping VLM tests`);
      return;
    }

    pipeline = await VLMPipeline(MODEL_PATH, "CPU");
    pipeline.setGenerationConfig({ max_new_tokens: 10 });
  });

  it("should generate text without images", async () => {
    const result = await pipeline.generate("What is 2+2?");

    assert.ok(result instanceof DecodedResults, "Result should be instance of DecodedResults");
    assert.ok(
      result instanceof VLMDecodedResults,
      "Result should be instance of VLMDecodedResults",
    );
    assert.ok(result.texts.length > 0, "Should generate some output");
  });

  it("should generate text with images", async () => {
    const testImage1 = createTestImageTensor();
    const testImage2 = createTestImageTensor();
    const result = await pipeline.generate("Compare these two images.", {
      images: [testImage1, testImage2],
    });

    assert.strictEqual(result.texts.length, 1, "Should generate comparison");
  });

  it("should generate text with video input", async () => {
    const testVideo = createTestVideoTensor();

    const result = await pipeline.generate("Describe what happens in this video.", {
      images: [],
      videos: [testVideo],
      generationConfig: {
        max_new_tokens: 20,
        temperature: 0,
      },
    });

    assert.strictEqual(result.texts.length, 1);
  });

  it("should generate with both image and video", async () => {
    const testImage = createTestImageTensor();
    const testVideo = createTestVideoTensor();

    const result = await pipeline.generate("Compare the image and video.", {
      images: [testImage],
      videos: [testVideo],
      generationConfig: { max_new_tokens: 20, temperature: 0 },
    });

    assert.strictEqual(result.texts.length, 1);
  });

  it("throw error on invalid streamer", async () => {
    await assert.rejects(
      pipeline.generate("What is 2+2?", {
        streamer: () => {
          throw new Error("Test error");
        },
      }),
      /Test error/,
    );
  });

  it("throw error with invalid generationConfig", async () => {
    await assert.rejects(
      pipeline.generate("What is 2+2?", {
        generationConfig: { max_new_tokens: "five" },
      }),
      /vlmPerformInferenceThread error/,
    );
  });

  it("should support streaming generation", async () => {
    const testImage = createTestImageTensor();
    const chunks = [];

    const stream = pipeline.stream("What do you see?", {
      images: [testImage],
      generationConfig: {
        max_new_tokens: 15,
        temperature: 0,
      },
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    assert.ok(chunks.length > 0, "Should receive streaming chunks");
    const fullOutput = chunks.join("");
    assert.ok(fullOutput.length > 0, "Combined chunks should form output");
  });

  it("should return VLMDecodedResults with perfMetrics", async () => {
    const testImage = createTestImageTensor();
    const result = await pipeline.generate("Describe the image.", {
      images: [testImage],
      generationConfig: {
        max_new_tokens: 10,
        temperature: 0,
      },
    });

    assert.ok(result, "Should return result");
    assert.ok(result.perfMetrics, "Should have perfMetrics");
    // Property from base PerformanceMetrics
    const numTokens = result.perfMetrics.getNumGeneratedTokens();
    assert.ok(typeof numTokens === "number", "getNumGeneratedTokens should return number");
    assert.ok(numTokens > 0, "Should generate at least one token");
    // VLM-specific properties
    const prepareEmbeddings = result.perfMetrics.getPrepareEmbeddingsDuration();
    assert.ok(
      typeof prepareEmbeddings.mean === "number",
      "PrepareEmbeddingsDuration should have mean",
    );
    const { prepareEmbeddingsDurations } = result.perfMetrics.vlmRawMetrics;
    assert.ok(
      Array.isArray(prepareEmbeddingsDurations),
      "Should have duration of preparation of embeddings",
    );
    assert.ok(prepareEmbeddingsDurations.length > 0, "Should have at least one duration value");
  });

  it("should get tokenizer from pipeline", () => {
    const tokenizer = pipeline.getTokenizer();
    assert.ok(tokenizer instanceof Tokenizer, "Should return tokenizer");
  });

  it("should start and finish chat", async () => {
    await pipeline.startChat("You are an assistant named Tom.");
    const result1 = await pipeline.generate("What is your name?");
    assert.ok(/Tom/.test(result1.toString()));

    await pipeline.finishChat();
    const result2 = await pipeline.generate("What is your name?");
    assert.ok(!/Tom/.test(result2.toString()));
  });
});
