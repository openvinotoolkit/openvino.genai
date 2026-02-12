// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert";
import { describe, it, mock } from "node:test";

// Mock the addon module to avoid requiring native binaries.
// The mock provides a minimal Text2VideoPipelineWrapper that simulates
// the C++ backend's init/generate interface.
mock.module("../dist/addon.js", {
    namedExports: {
        Text2VideoPipeline: class MockText2VideoPipelineWrapper {
            init(modelPath, device, properties, callback) {
                if (modelPath === "non_existent_path") {
                    callback(new Error("Failed to open non_existent_path"));
                    return;
                }
                callback(null);
            }
            generate(prompt, callback, config) {
                // Simulate returning a mock tensor result
                const mockTensor = {
                    data: new Float32Array([1.0, 2.0, 3.0]),
                    getShape: () => [1, 3],
                    getElementType: () => "f32",
                };
                callback(null, mockTensor);
            }
        },
    },
});

// Import after mock is set up
const { Text2VideoPipeline } = await import("../dist/pipelines/text2VideoPipeline.js");

describe("Text2VideoPipeline", () => {
    it("should be able to instantiate the pipeline", () => {
        const pipeline = new Text2VideoPipeline("dummy_path", "CPU");
        assert.ok(pipeline);
        assert.strictEqual(typeof pipeline.init, "function");
        assert.strictEqual(typeof pipeline.generate, "function");
    });

    it("should initialize successfully with a mock model", async () => {
        const pipeline = new Text2VideoPipeline("valid_model_path", "CPU");
        await pipeline.init();
        // If init doesn't throw, the pipeline is initialized
        assert.ok(true);
    });

    it("should fail if initialized without a real model", async () => {
        const pipeline = new Text2VideoPipeline("non_existent_path", "CPU");
        await assert.rejects(
            async () => {
                await pipeline.init();
            },
            {
                message: /Failed to open/,
            },
        );
    });

    it("should throw if generate is called before init", async () => {
        const pipeline = new Text2VideoPipeline("dummy_path", "CPU");
        await assert.rejects(
            async () => {
                await pipeline.generate("a cat");
            },
            {
                message: /Pipeline is not initialized/,
            },
        );
    });

    it("should generate a result after initialization", async () => {
        const pipeline = new Text2VideoPipeline("valid_model_path", "CPU");
        await pipeline.init();
        const result = await pipeline.generate("a flying cat", {});
        assert.ok(result);
        assert.ok(result.data);
        assert.deepStrictEqual(result.getShape(), [1, 3]);
    });
});
