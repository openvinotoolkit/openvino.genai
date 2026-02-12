// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text2VideoPipeline } from "openvino-genai-node";

/**
 * Basic sample for text-to-video generation using OpenVINO GenAI Node.js bindings.
 */
async function main() {
    if (process.argv.length < 3) {
        console.log("Usage: node text2video.js <model_path> [device]");
        process.exit(1);
    }

    const modelPath = process.argv[2];
    const device = process.argv[3] || "CPU";
    const prompt = "A futuristic city under a neon sunset, slow camera pan.";

    console.log(`--- Initializing Text2VideoPipeline for ${device} ---`);
    const pipeline = new Text2VideoPipeline(modelPath, device);
    await pipeline.init();

    console.log(`--- Generating Video for prompt: "${prompt}" ---`);
    // Note: Video generation can take several minutes on CPU
    const videoTensor = await pipeline.generate(prompt, {
        width: 480,
        height: 270,
        num_frames: 16,
    });

    console.log("--- Generation Complete ---");
    console.log("Resulting Tensor Shape:", videoTensor.getShape());
    // In a real application, you would save these frames to a file (e.g., .mp4 or .gif)
}

main().catch(err => {
    console.error("An error occurred:", err);
});
