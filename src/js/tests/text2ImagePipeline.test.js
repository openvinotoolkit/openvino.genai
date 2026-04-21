// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { Text2ImagePipeline } from "../dist/index.js";
import { Text2ImagePipeline as Text2ImagePipelineClass } from "../dist/pipelines/text2ImagePipeline.js";

const { IMAGE_GENERATION_MODEL_PATH } = process.env;

if (!IMAGE_GENERATION_MODEL_PATH) {
  throw new Error(
    "Environment variable IMAGE_GENERATION_MODEL_PATH must be set to the image generation model directory for tests.",
  );
}

describe("Text2ImagePipeline creation", () => {
  it("Text2ImagePipeline(modelPath, device) creates and initializes pipeline", async () => {
    const pipeline = await Text2ImagePipeline(IMAGE_GENERATION_MODEL_PATH, "CPU");
    assert.ok(pipeline);
    assert.strictEqual(typeof pipeline.generate, "function");
    assert.strictEqual(typeof pipeline.getPerformanceMetrics, "function");
    assert.strictEqual(typeof pipeline.getGenerationConfig, "function");
    assert.strictEqual(typeof pipeline.setGenerationConfig, "function");
  });
});

describe("Text2ImagePipeline methods", () => {
  let pipeline;

  before(async () => {
    pipeline = await Text2ImagePipeline(IMAGE_GENERATION_MODEL_PATH, "CPU");
  });

  it("generate(prompt) returns image tensor", async () => {
    const result = await pipeline.generate("a tiny robot", {
      width: 64,
      height: 64,
      num_inference_steps: 2,
    });

    assert.ok(result);
    assert.ok(result.data instanceof Uint8Array);
    assert.ok(result.data.length > 0);
    assert.deepStrictEqual(result.getShape(), [1, 64, 64, 3]);
  });

  it("generate(prompt, generationConfig) supports num_images_per_prompt", async () => {
    const result = await pipeline.generate("a tiny robot", {
      width: 64,
      height: 64,
      num_inference_steps: 2,
      num_images_per_prompt: 2,
    });

    assert.ok(result.data instanceof Uint8Array);
    assert.ok(result.data.length > 0);
    assert.deepStrictEqual(result.getShape(), [2, 64, 64, 3]);
  });

  it("getGenerationConfig() returns config object", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(config && typeof config === "object");
    assert.strictEqual(typeof config.guidance_scale, "number");
    assert.strictEqual(typeof config.num_inference_steps, "number");
    assert.strictEqual(typeof config.strength, "number");
  });

  it("setGenerationConfig(config) updates config", () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig({ guidance_scale: 3.5, num_inference_steps: 2 });
    const updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.guidance_scale, 3.5);
    assert.strictEqual(updated.num_inference_steps, 2);
    pipeline.setGenerationConfig(original);
  });

  it("getPerformanceMetrics() exposes image-generation-specific getters", async () => {
    await pipeline.generate("a tiny robot", {
      width: 64,
      height: 64,
      num_inference_steps: 2,
    });
    const pm = pipeline.getPerformanceMetrics();
    assert.ok(pm);
    assert.strictEqual(typeof pm.getLoadTime(), "number");
    assert.strictEqual(typeof pm.getGenerateDuration(), "number");
    assert.strictEqual(typeof pm.getInferenceDuration(), "number");

    const iterationDuration = pm.getIterationDuration();
    assert.ok(iterationDuration && typeof iterationDuration === "object");
    assert.strictEqual(typeof iterationDuration.mean, "number");
    assert.strictEqual(typeof iterationDuration.std, "number");

    const raw = pm.rawMetrics;
    assert.ok(raw);
    assert.ok(Array.isArray(raw.iterationDurations));
    assert.ok(Array.isArray(raw.unetInferenceDurations));
    assert.ok(Array.isArray(raw.transformerInferenceDurations));
  });

  it("setGenerationConfig(config) rejects undefined config", () => {
    assert.throws(() => pipeline.setGenerationConfig(undefined), /cannot be undefined or null/i);
  });

  it("generate calls callback on each step", async () => {
    const steps = [];
    const result = await pipeline.generate("a tiny robot", {
      width: 64,
      height: 64,
      num_inference_steps: 3,
      callback: (step, numSteps) => {
        steps.push({ step, numSteps });
        assert.strictEqual(typeof step, "number");
        assert.strictEqual(typeof numSteps, "number");
        return false;
      },
    });
    assert.ok(steps.length > 0, "callback should be called at least once");
    assert.strictEqual(steps[0].numSteps, 3);
    assert.ok(result.data.length > 0);
  });

  it("generate stops early when callback returns true", async () => {
    const steps = [];
    await pipeline.generate("a tiny robot", {
      width: 64,
      height: 64,
      num_inference_steps: 5,
      callback: (step) => {
        steps.push(step);
        return step >= 1; // stop after step 1
      },
    });
    assert.ok(steps.length <= 3, "Should stop early");
  });
});

describe("Text2ImagePipeline initialization", () => {
  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new Text2ImagePipelineClass(IMAGE_GENERATION_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.generate("a tiny robot"),
      /Text2ImagePipeline is not initialized/,
    );
  });
});
