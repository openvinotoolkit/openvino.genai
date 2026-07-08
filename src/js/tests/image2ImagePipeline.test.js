// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import os from "node:os";
import assert from "node:assert/strict";
import { addon as ov } from "openvino-node";
import { Image2ImagePipeline } from "../dist/index.js";
import { Image2ImagePipeline as Image2ImagePipelineClass } from "../dist/pipelines/image2ImagePipeline.js";
import { createTestImageTensor } from "./utils.js";

const { IMAGE_GENERATION_MODEL_PATH } = process.env;

if (!IMAGE_GENERATION_MODEL_PATH) {
  throw new Error(
    "Environment variable IMAGE_GENERATION_MODEL_PATH must be set to the image generation model directory for tests.",
  );
}

describe("Image2ImagePipeline creation", () => {
  it("Image2ImagePipeline(modelPath, device) creates and initializes pipeline", async () => {
    const pipeline = await Image2ImagePipeline(IMAGE_GENERATION_MODEL_PATH, "CPU");
    assert.ok(pipeline);
    assert.strictEqual(typeof pipeline.generate, "function");
    assert.strictEqual(typeof pipeline.getPerformanceMetrics, "function");
    assert.strictEqual(typeof pipeline.getGenerationConfig, "function");
    assert.strictEqual(typeof pipeline.setGenerationConfig, "function");
  });
});

// Skip due to CVS-179949
describe("Image2ImagePipeline methods", { skip: os.platform() === "darwin" }, () => {
  let pipeline;
  let testImage;

  before(async () => {
    pipeline = await Image2ImagePipeline(IMAGE_GENERATION_MODEL_PATH, "CPU");
    testImage = createTestImageTensor(64, 64);
    // The image2image pipeline expects a batch dimension, even for single images.
    // The test utility creates a [height, width, channels] tensor, so we need to
    // reshape it to [1, height, width, channels] before passing it to the pipeline.
    testImage.setShape([1, 64, 64, 3]);
  });

  it("generate(prompt, image) returns image tensor", async () => {
    const result = await pipeline.generate("a tiny robot", testImage);

    assert.ok(result);
    assert.ok(result.data instanceof Uint8Array);
    assert.ok(result.data.length > 0);
  });

  it("generate(prompt, image, generationConfig) honors width/height overrides", async () => {
    const result = await pipeline.generate("a tiny robot", testImage, {
      width: 128,
      height: 128,
    });

    assert.ok(result.data instanceof Uint8Array);
    assert.ok(result.data.length > 0);
    assert.deepStrictEqual(result.getShape(), [1, 128, 128, 3]);
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
    pipeline.setGenerationConfig({ guidance_scale: 3.5, num_inference_steps: 2, strength: 0.5 });
    const updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.guidance_scale, 3.5);
    assert.strictEqual(updated.num_inference_steps, 2);
    assert.strictEqual(updated.strength, 0.5);
    pipeline.setGenerationConfig(original);
  });

  it("getPerformanceMetrics() exposes image-generation-specific getters", async () => {
    await pipeline.generate("a tiny robot", testImage);
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
    const result = await pipeline.generate("a tiny robot", testImage, {
      width: 64,
      height: 64,
      num_inference_steps: 5,
      // Set strength to 1.0 to generate for all steps
      strength: 1.0,
      callback: (step, numSteps) => {
        steps.push({ step, numSteps });
        assert.strictEqual(typeof step, "number");
        assert.strictEqual(typeof numSteps, "number");
        return false;
      },
    });
    assert.strictEqual(steps.length, 5, "callback should be called at every step");
    assert.strictEqual(steps[0].numSteps, 5);
    assert.ok(result.data.length > 0);
  });

  it("generate stops early when callback returns true", async () => {
    const steps = [];
    await pipeline.generate("a tiny robot", testImage, {
      width: 64,
      height: 64,
      num_inference_steps: 5,
      // Set strength to 1.0 to generate for all steps
      strength: 1.0,
      callback: (step) => {
        steps.push(step);
        return step === 1;
      },
    });
    assert.strictEqual(steps.length, 2, "Should stop after steps 0 and 1");
  });

  it("decode(latent) returns an image tensor from a callback latent", async () => {
    let decoded;
    await pipeline.generate("a tiny robot", testImage, {
      width: 64,
      height: 64,
      num_inference_steps: 2,
      strength: 1.0,
      callback: async (step, numSteps, latent) => {
        assert.ok(latent, "callback should receive a latent tensor");
        decoded = await pipeline.decode(latent);
        return true; // stop early, we only need one latent
      },
    });
    assert.ok(decoded, "should have captured a decoded image");
    assert.deepStrictEqual(decoded.getShape(), [1, 64, 64, 3]);
    assert.ok(decoded.data instanceof Uint8Array);
    assert.ok(decoded.data.length > 0);
  });

  it("generate(prompt, image) rejects unbatched rank-3 tensor", async () => {
    const unbatchedImage = createTestImageTensor(64, 64);
    assert.strictEqual(unbatchedImage.getShape().length, 3);
    await assert.rejects(
      pipeline.generate("a tiny robot", unbatchedImage),
      /batched NHWC shape \[1, H, W, 3\]/,
    );
  });

  it("generate(prompt, image) rejects non-u8 image tensor", async () => {
    const floatImage = new ov.Tensor("f32", [1, 64, 64, 3], new Float32Array(64 * 64 * 3));
    await assert.rejects(pipeline.generate("a tiny robot", floatImage), /u8 element type/);
  });
});

// Skip due to CVS-179949
describe("Image2ImagePipeline concurrency", { skip: os.platform() === "darwin" }, () => {
  let pipeline;
  let testImage;
  let latent;

  before(async () => {
    pipeline = await Image2ImagePipeline(IMAGE_GENERATION_MODEL_PATH, "CPU");
    testImage = createTestImageTensor(64, 64);
    testImage.setShape([1, 64, 64, 3]);
    await pipeline.generate("a tiny robot", testImage, {
      width: 64,
      height: 64,
      num_inference_steps: 2,
      strength: 1.0,
      callback: (step, numSteps, capturedLatent) => {
        latent = capturedLatent;
        return true; // stop early, we only need one latent
      },
    });
  });

  it("decode() rejects while a generate() is in progress", async () => {
    const generating = pipeline.generate("a tiny robot", testImage, {
      width: 64,
      height: 64,
      num_inference_steps: 2,
      strength: 1.0,
    });
    await assert.rejects(
      pipeline.decode(latent),
      /decode\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await generating;
  });

  it("generate() rejects while a decode() is in progress", async () => {
    const decoding = pipeline.decode(latent);
    await assert.rejects(
      pipeline.generate("a tiny robot", testImage, {
        width: 64,
        height: 64,
        num_inference_steps: 2,
        strength: 1.0,
      }),
      /generate\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await decoding;
  });
});

describe("Image2ImagePipeline initialization", () => {
  let testImage;

  before(async () => {
    testImage = createTestImageTensor(64, 64);
    testImage.setShape([1, 64, 64, 3]);
  });

  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new Image2ImagePipelineClass(IMAGE_GENERATION_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.generate("a tiny robot", testImage),
      /Image2ImagePipeline is not initialized/,
    );
  });

  it("throws when decode() is called before init()", async () => {
    const uninitializedPipeline = new Image2ImagePipelineClass(IMAGE_GENERATION_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.decode(undefined),
      /Image2ImagePipeline is not initialized/,
    );
  });
});
