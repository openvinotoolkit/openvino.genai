// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import os from "node:os";
import assert from "node:assert/strict";
import { Text2VideoPipeline } from "../dist/index.js";
import { Text2VideoPipeline as Text2VideoPipelineClass } from "../dist/pipelines/text2VideoPipeline.js";

const { VIDEO_GENERATION_MODEL_PATH } = process.env;

if (!VIDEO_GENERATION_MODEL_PATH) {
  throw new Error(
    "Environment variable VIDEO_GENERATION_MODEL_PATH must be set to the video generation model directory for tests.",
  );
}

// Small sizes accepted by the tiny test model
const SMALL_VIDEO = { width: 32, height: 32, num_frames: 9, num_inference_steps: 2 };
// The pipeline runs the step callback on a worker thread and skips the callbacks still waiting when the
// generation ends. More steps leave time for the callbacks that the tests rely on.
const CALLBACK_VIDEO = { ...SMALL_VIDEO, num_inference_steps: 20 };

describe("Text2VideoPipeline creation", () => {
  it("Text2VideoPipeline(modelPath, device) creates and initializes pipeline", async () => {
    const pipeline = await Text2VideoPipeline(VIDEO_GENERATION_MODEL_PATH, "CPU");
    assert.ok(pipeline);
    assert.strictEqual(typeof pipeline.generate, "function");
    assert.strictEqual(typeof pipeline.decode, "function");
    assert.strictEqual(typeof pipeline.getGenerationConfig, "function");
    assert.strictEqual(typeof pipeline.setGenerationConfig, "function");
  });
});

describe("Text2VideoPipeline generation config", () => {
  let pipeline;

  before(async () => {
    pipeline = await Text2VideoPipeline(VIDEO_GENERATION_MODEL_PATH, "CPU");
  });

  it("getGenerationConfig() returns config object", () => {
    const config = pipeline.getGenerationConfig();
    assert.ok(config && typeof config === "object");
    assert.strictEqual(typeof config.guidance_scale, "number");
    assert.strictEqual(typeof config.num_inference_steps, "number");
    assert.strictEqual(typeof config.num_frames, "number");
    assert.strictEqual(typeof config.num_videos_per_prompt, "number");
    assert.strictEqual(typeof config.height, "number");
    assert.strictEqual(typeof config.width, "number");
  });

  it("setGenerationConfig(config) updates only the passed fields", () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig({ guidance_scale: 3.5, num_frames: 9 });
    const updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.guidance_scale, 3.5);
    assert.strictEqual(updated.num_frames, 9);
    assert.strictEqual(updated.height, original.height);
    assert.strictEqual(updated.width, original.width);
    assert.strictEqual(updated.num_inference_steps, original.num_inference_steps);
    pipeline.setGenerationConfig(original);
    assert.deepStrictEqual(pipeline.getGenerationConfig(), original);
  });

  it("setGenerationConfig(config) sets and clears optional fields", () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig({ negative_prompt: "blurry", frame_rate: 12 });
    let updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.negative_prompt, "blurry");
    assert.strictEqual(updated.frame_rate, 12);

    pipeline.setGenerationConfig({ negative_prompt: null });
    updated = pipeline.getGenerationConfig();
    assert.strictEqual(updated.negative_prompt, undefined);
    pipeline.setGenerationConfig(original);
  });

  it("setGenerationConfig(config) round-trips list fields", () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig({ spatio_temporal_guidance_blocks: [1, 2] });
    assert.deepStrictEqual(pipeline.getGenerationConfig().spatio_temporal_guidance_blocks, [1, 2]);
    pipeline.setGenerationConfig({ spatio_temporal_guidance_blocks: null });
    assert.strictEqual(pipeline.getGenerationConfig().spatio_temporal_guidance_blocks, undefined);
    pipeline.setGenerationConfig(original);
  });

  it("setGenerationConfig(config) rejects undefined config", () => {
    assert.throws(() => pipeline.setGenerationConfig(undefined), /cannot be undefined or null/i);
  });

  it("setGenerationConfig(config) rejects invalid values", () => {
    assert.throws(
      () => pipeline.setGenerationConfig({ num_frames: -1 }),
      /VideoGenerationConfig\.num_frames/,
    );
    assert.throws(
      () => pipeline.setGenerationConfig({ num_frames: 1.5 }),
      /VideoGenerationConfig\.num_frames/,
    );
    assert.throws(
      () => pipeline.setGenerationConfig({ guidance_scale: "high" }),
      /VideoGenerationConfig\.guidance_scale/,
    );
    // Out of the int range of max_sequence_length
    assert.throws(
      () => pipeline.setGenerationConfig({ max_sequence_length: 2 ** 32 + 128 }),
      /VideoGenerationConfig\.max_sequence_length/,
    );
    assert.throws(
      () => pipeline.setGenerationConfig({ num_images_per_prompt: 2 }),
      /Unsupported VideoGenerationConfig property: num_images_per_prompt/,
    );
  });

  // The options are validated before the generation starts
  it("generate(prompt, options) rejects invalid options", async () => {
    await assert.rejects(
      pipeline.generate("a tiny robot", { ...SMALL_VIDEO, num_frames: "nine" }),
      /VideoGenerationConfig\.num_frames/,
    );
    await assert.rejects(
      pipeline.generate("a tiny robot", { ...SMALL_VIDEO, num_images_per_prompt: 2 }),
      /Unsupported VideoGenerationConfig property: num_images_per_prompt/,
    );
    await assert.rejects(
      pipeline.generate(42),
      /Text2VideoPipeline\.generate\(\) expects a prompt string/,
    );
  });
});

// Skip due to CVS-179949
describe("Text2VideoPipeline methods", { skip: os.platform() === "darwin" }, () => {
  let pipeline;

  before(async () => {
    pipeline = await Text2VideoPipeline(VIDEO_GENERATION_MODEL_PATH, "CPU");
  });

  it("generate(prompt, options) returns a video tensor", async () => {
    const result = await pipeline.generate("a tiny robot", SMALL_VIDEO);

    assert.ok(result.video.data instanceof Uint8Array);
    assert.deepStrictEqual(result.video.getShape(), [1, 9, 32, 32, 3]);
    // The tiny LTX-Video model has no audio
    assert.strictEqual(result.audio, undefined);
    assert.strictEqual(result.audioSampleRate, 0);
  });

  it("generate(prompt, options) supports num_videos_per_prompt", async () => {
    // A separate pipeline, so that the batch size used here does not affect the other tests
    const batchPipeline = await Text2VideoPipeline(VIDEO_GENERATION_MODEL_PATH, "CPU");
    const result = await batchPipeline.generate("a tiny robot", {
      ...SMALL_VIDEO,
      num_videos_per_prompt: 2,
    });

    assert.deepStrictEqual(result.video.getShape(), [2, 9, 32, 32, 3]);
  });

  it("generate(prompt, options) returns video generation perf metrics", async () => {
    const { perfMetrics } = await pipeline.generate("a tiny robot", SMALL_VIDEO);
    assert.ok(perfMetrics);
    assert.strictEqual(typeof perfMetrics.getLoadTime(), "number");
    assert.ok(perfMetrics.getGenerateDuration() > 0);
    assert.strictEqual(typeof perfMetrics.getInferenceDuration(), "number");
    assert.strictEqual(typeof perfMetrics.getVaeDecoderInferDuration(), "number");

    const transformerDuration = perfMetrics.getTransformerInferDuration();
    assert.strictEqual(typeof transformerDuration.mean, "number");
    assert.strictEqual(typeof transformerDuration.std, "number");

    const raw = perfMetrics.rawMetrics;
    assert.ok(Array.isArray(raw.iterationDurations));
    assert.ok(Array.isArray(raw.transformerInferenceDurations));
  });

  it("generate uses the generation config set on the pipeline", async () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig(SMALL_VIDEO);
    try {
      const result = await pipeline.generate("a tiny robot");
      assert.deepStrictEqual(result.video.getShape(), [1, 9, 32, 32, 3]);
    } finally {
      pipeline.setGenerationConfig(original);
    }
  });

  it("generate options override the pipeline config for that call only", async () => {
    const original = pipeline.getGenerationConfig();
    pipeline.setGenerationConfig(SMALL_VIDEO);
    try {
      const result = await pipeline.generate("a tiny robot", { num_frames: 17 });
      assert.deepStrictEqual(result.video.getShape(), [1, 17, 32, 32, 3]);
      assert.strictEqual(pipeline.getGenerationConfig().num_frames, 9);
    } finally {
      pipeline.setGenerationConfig(original);
    }
  });

  it("generate calls callback on each step", async () => {
    const numSteps = CALLBACK_VIDEO.num_inference_steps;
    const steps = [];
    const result = await pipeline.generate("a tiny robot", {
      ...CALLBACK_VIDEO,
      callback: (step, numSteps) => {
        steps.push({ step, numSteps });
        return false;
      },
    });
    // The callbacks of the last steps can be skipped, the others come in order
    assert.ok(
      steps.length > 0 && steps.length <= numSteps,
      `unexpected number of callbacks: ${steps.length}`,
    );
    steps.forEach((call, index) => assert.deepStrictEqual(call, { step: index, numSteps }));
    assert.deepStrictEqual(result.video.getShape(), [1, 9, 32, 32, 3]);
  });

  it("callback receives a latent tensor that can be decoded", async () => {
    let decoded;
    await pipeline.generate("a tiny robot", {
      ...CALLBACK_VIDEO,
      callback: async (step, numSteps, latent) => {
        assert.strictEqual(typeof latent.getShape, "function");
        decoded = await pipeline.decode(latent);
        return true; // stop early, one decoded latent is enough
      },
    });
    assert.ok(decoded, "the callback should have decoded a latent");
    assert.deepStrictEqual(decoded.video.getShape(), [1, 9, 32, 32, 3]);
    assert.ok(decoded.video.data instanceof Uint8Array);
  });

  it("decode(latent) returns a video tensor", async () => {
    let captured;
    await pipeline.generate("a tiny robot", {
      ...CALLBACK_VIDEO,
      callback: (step, numSteps, latent) => {
        captured = latent;
        return true; // stop early, we only need one latent
      },
    });
    assert.ok(captured, "should have captured a latent");
    const decoded = await pipeline.decode(captured);
    assert.deepStrictEqual(decoded.video.getShape(), [1, 9, 32, 32, 3]);
    assert.strictEqual(decoded.audio, undefined);
  });

  it("generate stops early when the callback asks to stop", async () => {
    const steps = [];
    const result = await pipeline.generate("a tiny robot", {
      ...CALLBACK_VIDEO,
      callback: (step) => {
        steps.push(step);
        return true;
      },
    });
    assert.deepStrictEqual(steps, [0]);
    // A stopped generation returns an empty video, as in the C++ and Python API
    assert.deepStrictEqual(result.video.getShape(), []);
  });

  it("generate rejects when the callback throws", async () => {
    await assert.rejects(
      pipeline.generate("a tiny robot", {
        ...CALLBACK_VIDEO,
        callback: () => {
          throw new Error("callback failure");
        },
      }),
      /callback failure/,
    );
  });
});

// Skip due to CVS-179949
describe("Text2VideoPipeline concurrency", { skip: os.platform() === "darwin" }, () => {
  let pipeline;
  let latent;

  before(async () => {
    pipeline = await Text2VideoPipeline(VIDEO_GENERATION_MODEL_PATH, "CPU");
    await pipeline.generate("a tiny robot", {
      ...CALLBACK_VIDEO,
      callback: (step, numSteps, capturedLatent) => {
        latent = capturedLatent;
        return true; // stop early, we only need one latent
      },
    });
  });

  it("decode() rejects while a generate() is in progress", async () => {
    const generating = pipeline.generate("a tiny robot", SMALL_VIDEO);
    await assert.rejects(
      pipeline.decode(latent),
      /decode\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await generating;
  });

  it("generate() rejects while a decode() is in progress", async () => {
    const decoding = pipeline.decode(latent);
    await assert.rejects(
      pipeline.generate("a tiny robot", SMALL_VIDEO),
      /generate\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await decoding;
  });

  it("a rejected generate() does not release a decode() in progress", async () => {
    const decoding = pipeline.decode(latent);
    await assert.rejects(
      pipeline.generate("a tiny robot", SMALL_VIDEO),
      /generate\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await assert.rejects(
      pipeline.decode(latent),
      /decode\(\) cannot run while another generate\(\) or decode\(\) is in progress/,
    );
    await decoding;
  });

  it("setGenerationConfig() rejects while a generate() is in progress", async () => {
    const original = pipeline.getGenerationConfig();
    const generating = pipeline.generate("a tiny robot", SMALL_VIDEO);
    assert.throws(
      () => pipeline.setGenerationConfig({ num_frames: 17 }),
      /setGenerationConfig\(\) cannot run while generate\(\) is in progress/,
    );
    await generating;
    pipeline.setGenerationConfig({ num_frames: 17 });
    assert.strictEqual(pipeline.getGenerationConfig().num_frames, 17);
    pipeline.setGenerationConfig(original);
  });

  it("a rejected generate() call leaves the pipeline usable", async () => {
    await assert.rejects(pipeline.generate(42));
    const result = await pipeline.generate("a tiny robot", SMALL_VIDEO);
    assert.deepStrictEqual(result.video.getShape(), [1, 9, 32, 32, 3]);
  });
});

describe("Text2VideoPipeline initialization", () => {
  it("throws when generate() is called before init()", async () => {
    const uninitializedPipeline = new Text2VideoPipelineClass(VIDEO_GENERATION_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.generate("a tiny robot"),
      /Text2VideoPipeline is not initialized/,
    );
  });

  it("throws when decode() is called before init()", async () => {
    const uninitializedPipeline = new Text2VideoPipelineClass(VIDEO_GENERATION_MODEL_PATH, "CPU");
    await assert.rejects(
      uninitializedPipeline.decode(undefined),
      /Text2VideoPipeline is not initialized/,
    );
  });

  it("throws when the config is accessed before init()", () => {
    const uninitializedPipeline = new Text2VideoPipelineClass(VIDEO_GENERATION_MODEL_PATH, "CPU");
    assert.throws(
      () => uninitializedPipeline.getGenerationConfig(),
      /Text2VideoPipeline is not initialized/,
    );
    assert.throws(
      () => uninitializedPipeline.setGenerationConfig({ num_frames: 9 }),
      /Text2VideoPipeline is not initialized/,
    );
  });
});
