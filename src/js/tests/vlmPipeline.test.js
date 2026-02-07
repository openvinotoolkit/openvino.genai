// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Tokenizer, VLMPipeline, DecodedResults, VLMDecodedResults } from "../dist/index.js";

import assert from "node:assert";
import { describe, it, before } from "node:test";
import { models } from "./models.js";
import { createTestImageTensor, createTestVideoTensor } from "./utils.js";

const MODEL_PATH = process.env.VLM_MODEL_PATH || `./tests/models/${models.VLM.split("/")[1]}`;

// Skip tests on macOS due to insufficient memory
describe("VLMPipeline", { skip: process.platform === "darwin" }, () => {
  let pipeline, testImage1, testImage2, testVideo1, testVideo2;

  before(async () => {
    pipeline = await VLMPipeline(MODEL_PATH, "CPU");
    pipeline.setGenerationConfig({ max_new_tokens: 10 });
    testImage1 = createTestImageTensor();
    testImage2 = createTestImageTensor(50, 50);
    testVideo1 = createTestVideoTensor();
    testVideo2 = createTestVideoTensor(6, 64, 64);
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
    const result = await pipeline.generate("Compare these two images.", {
      images: [testImage1, testImage2],
    });

    assert.strictEqual(result.texts.length, 1, "Should generate comparison");
  });

  it("should generate text with video input", async () => {
    const result = await pipeline.generate("Describe what happens in this video.", {
      videos: [testVideo1],
      generationConfig: {
        max_new_tokens: 20,
        temperature: 0,
      },
    });

    assert.strictEqual(result.texts.length, 1);
  });

  it("should generate with both image and video", async () => {
    const result = await pipeline.generate("Compare the image and video.", {
      images: [testImage1],
      videos: [testVideo2],
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
    const chunks = [];

    const stream = pipeline.stream("What do you see?", {
      images: [testImage1],
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
    const result = await pipeline.generate("Describe the image.", {
      images: [testImage2],
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
    assert.ok(
      0 < numTokens && numTokens <= 10,
      "Number of tokens should be between 0 and max_new_tokens",
    );
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
