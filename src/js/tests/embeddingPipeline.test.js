// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, before } from "node:test";
import { EmbeddingPipeline, PoolingType } from "../dist/index.js";
import { isFloat32Array } from "util/types";
import assert from "node:assert/strict";
import { createTestImageTensor, createTestVideoTensor } from "./utils.js";

const { EMBEDDING_MODEL_PATH, MULTIMODAL_EMBEDDINGS_MODEL_PATH } = process.env;

if (!EMBEDDING_MODEL_PATH) {
  throw new Error("Please set EMBEDDING_MODEL_PATH environment variable to run the tests.");
}

if (!MULTIMODAL_EMBEDDINGS_MODEL_PATH) {
  throw new Error(
    "Please set MULTIMODAL_EMBEDDINGS_MODEL_PATH environment variable to run the tests.",
  );
}

describe(
  "EmbeddingPipeline text-only model",
  { skip: ["darwin", "win32"].includes(process.platform) },
  () => {
    let pipeline = null;

    before(async () => {
      pipeline = await EmbeddingPipeline(EMBEDDING_MODEL_PATH, "CPU");
    });

    it("embeds a single text and a batch of texts", async () => {
      for (const text of ["What is OpenVINO?", ["Hello", "World"]]) {
        const { embeddings } = await pipeline.embed(text);
        const shape = embeddings.getShape();
        assert.strictEqual(shape.length, 2);
        assert.strictEqual(shape[0], Array.isArray(text) ? text.length : 1);
        assert.ok(shape[1] > 0);
        assert.ok(isFloat32Array(embeddings.data));
        assert.strictEqual(embeddings.data.length, shape[0] * shape[1]);
      }
    });

    it("accepts subsequent calls after a worker error", async () => {
      await assert.rejects(
        pipeline.embed("test", { images: [createTestImageTensor()] }),
        /This model does not support image\/video input/,
      );
      const { embeddings } = await pipeline.embed("test");
      assert.strictEqual(embeddings.getShape()[0], 1);
    });
  },
);

// Skip due to CVS-179949 and CVS-179949
describe(
  "EmbeddingPipeline multimodal initialization",
  { skip: ["darwin", "win32"].includes(process.platform) },
  () => {
    it("initializes with default properties", async () => {
      const pipeline = await EmbeddingPipeline(MULTIMODAL_EMBEDDINGS_MODEL_PATH, "CPU");
      assert.ok(pipeline instanceof Object);
    });

    it("initializes with pooling_type and normalize properties", async () => {
      const pipeline = await EmbeddingPipeline(MULTIMODAL_EMBEDDINGS_MODEL_PATH, "CPU", {
        pooling_type: PoolingType.MEAN,
        normalize: false,
      });
      assert.ok(pipeline instanceof Object);
    });
  },
);

// Skip due to CVS-179949 and CVS-179949
describe(
  "EmbeddingPipeline multimodal functions",
  { skip: ["darwin", "win32"].includes(process.platform) },
  () => {
    let pipeline = null;

    before(async () => {
      pipeline = await EmbeddingPipeline(MULTIMODAL_EMBEDDINGS_MODEL_PATH, "CPU");
    });

    it("async embed a single image", async () => {
      const { embeddings } = await pipeline.embed("", {
        images: [createTestImageTensor()],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single image with accompanying text", async () => {
      const { embeddings } = await pipeline.embed("Describe this image.", {
        images: [createTestImageTensor()],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single image with embedding_prompt", async () => {
      const { embeddings } = await pipeline.embed("Describe this image.", {
        images: [createTestImageTensor()],
        embedding_prompt: "Represent the user's input.",
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed multiple images in a single call", async () => {
      const { embeddings } = await pipeline.embed("", {
        images: [createTestImageTensor(), createTestImageTensor()],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single video with metadata", async () => {
      const { embeddings } = await pipeline.embed("Represent this video.", {
        videos: [createTestVideoTensor()],
        videosMetadata: [{ fps: 2.0, frames_indices: [0, 1, 2, 3] }],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single video without metadata", async () => {
      const { embeddings } = await pipeline.embed("Represent this video.", {
        videos: [createTestVideoTensor()],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single video with fps-only metadata", async () => {
      const { embeddings } = await pipeline.embed("Represent this video.", {
        videos: [createTestVideoTensor()],
        videosMetadata: [{ fps: 2.0 }],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed multiple videos in a single call", async () => {
      const { embeddings } = await pipeline.embed("Represent these videos.", {
        videos: [createTestVideoTensor(), createTestVideoTensor()],
        videosMetadata: [{ frames_indices: [0, 1, 2, 3] }, { frames_indices: [0, 1, 2, 3] }],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed an image and a video together", async () => {
      const { embeddings } = await pipeline.embed("Represent this.", {
        images: [createTestImageTensor()],
        videos: [createTestVideoTensor()],
        videosMetadata: [{ frames_indices: [0, 1, 2, 3] }],
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a single text", async () => {
      const { embeddings } = await pipeline.embed("What is OpenVINO?");
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed a batch of texts", async () => {
      const texts = ["What is OpenVINO?", "Represent this text."];
      const { embeddings } = await pipeline.embed(texts);
      assert.strictEqual(embeddings.getShape()[0], texts.length);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed text with embedding_prompt", async () => {
      const { embeddings } = await pipeline.embed("What is OpenVINO?", {
        embedding_prompt: "Represent the user's input.",
      });
      assert.strictEqual(embeddings.getShape()[0], 1);
      assert.ok(isFloat32Array(embeddings.data));
      assert.ok(embeddings.data.length > 0);
    });

    it("async embed batch of texts with image and video", async () => {
      const texts = ["Represent OpenVINO.", "Represent this image.", "Represent this video."];
      const { embeddings } = await pipeline.embed(texts, {
        images: [createTestImageTensor()],
        videos: [createTestVideoTensor()],
        videosMetadata: [{ frames_indices: [0, 1, 2, 3] }],
        embedding_prompt: "Represent the user's input.",
      });
      assert.strictEqual(embeddings.getShape()[0], texts.length);
    });
  },
);

// Skip due to CVS-179949 and CVS-179949
describe(
  "EmbeddingPipeline multimodal normalization",
  { skip: ["darwin", "win32"].includes(process.platform) },
  () => {
    it("normalize=true produces unit-norm embeddings", async () => {
      const pipeline = await EmbeddingPipeline(MULTIMODAL_EMBEDDINGS_MODEL_PATH, "CPU", {
        normalize: true,
      });
      const { embeddings } = await pipeline.embed("", {
        images: [createTestImageTensor()],
      });
      let sumOfSquares = 0;
      for (const value of embeddings.data) {
        sumOfSquares += value * value;
      }
      assert.ok(Math.abs(Math.sqrt(sumOfSquares) - 1.0) < 1e-3);
    });
  },
);
