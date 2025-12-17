import { bootstrap } from "global-agent";
import { promises as fs } from "node:fs";
import { listFiles, downloadFile } from "@huggingface/hub";
import { addon as ov } from "openvino-node";

const BASE_DIR = "./tests/models/";

bootstrap();

export async function downloadModel(repo) {
  console.log(`Downloading model '${repo}'`);

  const fetch = await import("node-fetch");
  const modelName = repo.split("/")[1];
  const destDir = `${BASE_DIR}${modelName}`;

  await fs.mkdir(destDir, { recursive: true });

  const fileList = await listFiles({
    repo,
    fetch: fetch.default,
  });
  const fileNames = [];
  for await (const file of fileList) {
    fileNames.push(file.path);
  }

  for (const path of fileNames) {
    console.log(`Downloading file '${path}'`);
    const response = await downloadFile({
      repo,
      path,
      fetch: fetch.default,
    });
    const filename = `${destDir}/${path}`;

    await saveFile(filename, response);
    console.log(`File '${path}' downloaded`);
  }

  console.log(`Model '${repo}' downloaded`);
}

async function saveFile(file, response) {
  const arrayBuffer = await response.arrayBuffer();

  await fs.writeFile(file, Buffer.from(arrayBuffer));
}

/**
 * Creates a synthetic test image tensor with a gradient pattern.
 *
 * Generates a small RGB image filled with a gradient pattern for testing VLM pipelines.
 * The red channel varies by height, green by width, and blue is constant.
 *
 * @param height - Height of the image in pixels. (default: 32)
 * @param width - Width of the image in pixels. (default: 32)
 * @returns An OpenVINO Tensor with shape [height, width, channels] and uint8 data type.
 */
export function createTestImageTensor(height = 32, width = 32) {
  const channels = 3;
  const data = new Uint8Array(height * width * channels);

  // Fill with gradient pattern
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const idx = (h * width + w) * channels;
      data[idx] = h * 8; // R
      data[idx + 1] = w * 8; // G
      data[idx + 2] = 128; // B
    }
  }

  return new ov.Tensor("u8", [height, width, channels], data);
}

/**
 * Creates a synthetic test video tensor with multiple frames.
 *
 * Generates a video tensor with a synthetic pattern that varies across frames.
 * Each frame has a slightly different color pattern to simulate temporal variation.
 * Useful for testing VLM pipelines with video inputs.
 *
 * @param frames - Number of video frames to generate. (default: 4)
 * @param height - Height of each frame in pixels. (default: 32)
 * @param width - Width of each frame in pixels. (default: 32)
 * @returns An OpenVINO Tensor with shape [frames, height, width, channels] and uint8 data type.
 */
export function createTestVideoTensor(frames = 4, height = 32, width = 32) {
  const channels = 3;
  const data = new Uint8Array(frames * height * width * channels);

  for (let f = 0; f < frames; f++) {
    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const idx = (f * height * width + h * width + w) * channels;
        data[idx] = (h + f * 10) % 256;
        data[idx + 1] = (w + f * 10) % 256;
        data[idx + 2] = 128;
      }
    }
  }

  return new ov.Tensor("u8", [frames, height, width, channels], data);
}
