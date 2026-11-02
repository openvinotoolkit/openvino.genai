// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename, resolve } from "node:path";
import { EmbeddingPipeline } from "openvino-genai-node";
import { hideBin } from "yargs/helpers";
import yargs from "yargs/yargs";
import { readImage } from "../image_utils.js";
import { readVideo } from "../video_utils.js";

/**
 * Computes cosine similarity between two embedding vectors.
 * @param {Float32Array|number[]} lhs - First embedding vector.
 * @param {Float32Array|number[]} rhs - Second embedding vector.
 * @returns {number} Cosine similarity in the range [-1, 1].
 */
function cosineSimilarity(lhs, rhs) {
  let dot = 0;
  let lhsNorm = 0;
  let rhsNorm = 0;
  for (let i = 0; i < lhs.length; i++) {
    dot += lhs[i] * rhs[i];
    lhsNorm += lhs[i] * lhs[i];
    rhsNorm += rhs[i] * rhs[i];
  }
  if (lhsNorm === 0 || rhsNorm === 0) {
    return 0;
  }
  return dot / (Math.sqrt(lhsNorm) * Math.sqrt(rhsNorm));
}

/**
 * Embeds a single image file.
 * @param {EmbeddingPipeline} pipeline - Initialized multimodal embedding pipeline.
 * @param {string} path - Path to the image file.
 * @returns {Promise<Float32Array>} Image embedding vector.
 */
async function embedImage(pipeline, path) {
  const imageTensor = await readImage(path);
  const { embeddings } = await pipeline.embed("", { images: [imageTensor] });
  return embeddings.data;
}

/**
 * Embeds a single video file.
 * @param {EmbeddingPipeline} pipeline - Initialized multimodal embedding pipeline.
 * @param {string} path - Path to the video file.
 * @param {number} numFrames - Number of frames to sample.
 * @returns {Promise<Float32Array>} Video embedding vector.
 */
async function embedVideo(pipeline, path, numFrames) {
  const { videoTensor, videoMetadata } = readVideo(path, numFrames);
  const { embeddings } = await pipeline.embed("", {
    videos: [videoTensor],
    videosMetadata: [videoMetadata],
  });
  return embeddings.data;
}

async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command("$0 <model_dir>", "Rank images and videos by similarity to a text query", (builder) =>
      builder.positional("model_dir", {
        type: "string",
        describe: "Path to the multimodal embedding model directory",
        demandOption: true,
      }),
    )
    .option("query", {
      type: "string",
      describe: "Text query used to find the most similar image or video",
      demandOption: true,
    })
    .option("images", {
      type: "array",
      default: [],
      describe: "Image paths to compare with the query",
    })
    .option("videos", {
      type: "array",
      default: [],
      describe: "Video paths to compare with the query",
    })
    .option("num-video-frames", {
      type: "number",
      default: 8,
      describe: "Number of video frames to sample",
    })
    .option("device", {
      type: "string",
      default: "CPU",
      describe: "Device to run the model on",
    })
    .strict()
    .help()
    .parse();

  const { model_dir: modelDir, query, images, videos, numVideoFrames, device } = argv;

  if (images.length === 0 && videos.length === 0) {
    throw new Error("At least one input must be provided via --images or --videos");
  }

  const pipeline = await EmbeddingPipeline(modelDir, device);
  const { embeddings: queryEmbeddings } = await pipeline.embed(query);
  const queryVector = queryEmbeddings.data;

  const results = [];
  for (const imagePath of images) {
    const imageVector = await embedImage(pipeline, String(imagePath));
    results.push({ score: cosineSimilarity(queryVector, imageVector), type: "image", path: String(imagePath) });
  }
  for (const videoPath of videos) {
    const videoVector = await embedVideo(pipeline, String(videoPath), numVideoFrames);
    results.push({ score: cosineSimilarity(queryVector, videoVector), type: "video", path: String(videoPath) });
  }

  results.sort((lhs, rhs) => rhs.score - lhs.score);

  console.log("Query:", query);
  console.log("Ranked inputs by cosine similarity:");
  results.forEach(({ score, type, path }, index) => {
    console.log(`${index + 1}. ${type}: ${resolve(path)} similarity=${score.toFixed(6)}`);
  });
  const best = results[0];
  console.log("Most similar input:", best.type, resolve(best.path), `similarity=${best.score.toFixed(6)}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
