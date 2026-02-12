// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert";
import { describe, it, before } from "node:test";
import { Text2VideoPipeline } from "../dist/index.js";

const { T2V_PATH } = process.env;

if (!T2V_PATH) {
  throw new Error("Please set T2V_PATH environment variable to run the tests.");
}

describe("Text2VideoPipeline", () => {
  let pipeline;

  before(async () => {
    pipeline = await Text2VideoPipeline(T2V_PATH, "CPU");
  });

  it("should generate video from a text prompt", async () => {
    const result = await pipeline.generate("A scenic landscape", {
      num_inference_steps: 2,
      height: 64,
      width: 64,
      num_frames: 9,
    });

    assert.ok(result, "Should return a result");
    assert.ok(result.video, "Result should have a video tensor");
    assert.ok(result.perfMetrics, "Result should have perfMetrics");
  });

  it("should return performance metrics", async () => {
    const result = await pipeline.generate("A sunset over the ocean", {
      num_inference_steps: 2,
      height: 64,
      width: 64,
      num_frames: 9,
    });

    const metrics = result.perfMetrics;
    assert.ok(typeof metrics.loadTime === "number", "loadTime should be a number");
    assert.ok(typeof metrics.generateDuration === "number", "generateDuration should be a number");
    assert.ok(typeof metrics.iterationDuration.mean === "number", "iterationDuration.mean should be a number");
    assert.ok(typeof metrics.iterationDuration.std === "number", "iterationDuration.std should be a number");
    assert.ok(typeof metrics.transformerInferDuration.mean === "number", "transformerInferDuration.mean should be a number");
    assert.ok(typeof metrics.vaeDecoderInferDuration === "number", "vaeDecoderInferDuration should be a number");
  });

  it("should get generation config", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(typeof config === "object", "Config should be an object");
    assert.ok(typeof config.guidance_scale === "number", "guidance_scale should be a number");
    assert.ok(typeof config.height === "number", "height should be a number");
    assert.ok(typeof config.width === "number", "width should be a number");
  });

  it("should set generation config", () => {
    const originalConfig = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig({ guidance_scale: 5.0 });
    const updatedConfig = pipeline.getGenerationConfig();
    assert.strictEqual(updatedConfig.guidance_scale, 5.0, "guidance_scale should be updated");

    // Restore original config
    pipeline.setGenerationConfig(originalConfig);
  });

  it("should generate video with negative prompt", async () => {
    const result = await pipeline.generate("A cat playing", {
      negative_prompt: "blurry, low quality",
      num_inference_steps: 2,
      height: 64,
      width: 64,
      num_frames: 9,
    });

    assert.ok(result, "Should return a result");
    assert.ok(result.video, "Result should have a video tensor");
  });
});
