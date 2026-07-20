// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readFile, stat, readdir, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { addon as ov } from "openvino-node";
import bmp from "bmp-js";
import jpegJs from "jpeg-js";
import { PNG } from "pngjs";

const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp"]);

/**
 * Decodes the raw file bytes into a flat pixel buffer and exposes its layout.
 * @param {Buffer} buffer - Raw file contents.
 * @param {string} ext - File extension (including the leading dot).
 * @returns {{ width: number, height: number, data: Uint8Array, layout: "rgba" | "abgr" }}
 */
function decodeImage(buffer, ext) {
  const lower = ext.toLowerCase();
  if (lower === ".jpg" || lower === ".jpeg") {
    const { width, height, data } = jpegJs.decode(buffer, { useTArray: true });
    return { width, height, data, layout: "rgba" };
  }
  if (lower === ".png") {
    const { width, height, data } = PNG.sync.read(buffer);
    return { width, height, data, layout: "rgba" };
  }
  if (lower === ".bmp") {
    const { width, height, data } = bmp.decode(buffer);
    return { width, height, data, layout: "abgr" };
  }
  throw new Error(`Unsupported image format: ${ext}`);
}

/**
 * Converts the decoded pixel buffer into a packed HWC RGB Uint8Array,
 * equivalent to PIL's `Image.open(path).convert("RGB")`.
 * @param {{ width: number, height: number, data: Uint8Array, layout: "rgba" | "abgr" }} decoded
 * @returns {Uint8Array}
 */
function toRGB({ width, height, data, layout }) {
  const pixelCount = width * height;
  const rgb = new Uint8Array(pixelCount * 3);
  if (layout === "rgba") {
    for (let i = 0; i < pixelCount; i++) {
      rgb[i * 3] = data[i * 4];
      rgb[i * 3 + 1] = data[i * 4 + 1];
      rgb[i * 3 + 2] = data[i * 4 + 2];
    }
  } else if (layout === "abgr") {
    for (let i = 0; i < pixelCount; i++) {
      rgb[i * 3] = data[i * 4 + 3];
      rgb[i * 3 + 1] = data[i * 4 + 2];
      rgb[i * 3 + 2] = data[i * 4 + 1];
    }
  } else {
    throw new Error(`Unknown pixel layout: ${layout}`);
  }
  return rgb;
}

/**
 * Reads a JPEG/PNG/BMP image and returns an OpenVINO tensor in HWC RGB layout.
 *
 * @param {string} filePath - Path to a .jpg, .jpeg, .png or .bmp file.
 * @param {{ batched?: boolean }} [options]
 *   When `batched` is true, the returned tensor has shape `[1, H, W, 3]`
 *   (the layout expected by `Image2ImagePipeline` / `InpaintingPipeline`).
 *   Otherwise the shape is `[H, W, 3]` (the layout expected by `VLMPipeline`).
 * @returns {Promise<ov.Tensor>} `u8` tensor with RGB pixels.
 */
export async function readImage(filePath, { batched = false } = {}) {
  const buffer = await readFile(filePath);
  const decoded = decodeImage(buffer, extname(filePath));
  const rgb = toRGB(decoded);
  const shape = batched
    ? [1, decoded.height, decoded.width, 3]
    : [decoded.height, decoded.width, 3];

  const tensor = new ov.Tensor("u8", shape, rgb);
  tensor._buffer = rgb;
  return tensor;
}

/**
 * Reads one image file, or all supported images from a directory, and
 * returns them as a list of OpenVINO tensors sorted by filename.
 *
 * @param {string} path - File path or directory containing images.
 * @param {{ batched?: boolean }} [options] - See {@link readImage}.
 * @returns {Promise<ov.Tensor[]>}
 */
export async function readImages(path, options) {
  const stats = await stat(path);
  if (!stats.isDirectory()) {
    return [await readImage(path, options)];
  }

  const entries = await readdir(path, { withFileTypes: true });
  const files = entries
    .filter(
      (entry) =>
        entry.isFile() && SUPPORTED_EXTENSIONS.has(extname(entry.name).toLowerCase()),
    )
    .map((entry) => join(path, entry.name));
  files.sort((a, b) => a.localeCompare(b));

  const tensors = [];
  for (const file of files) {
    tensors.push(await readImage(file, options));
  }
  return tensors;
}

/**
 * Converts an RGB image tensor to ABGR buffer expected by `bmp-js` encoder.
 *
 * @param {ov.Tensor} tensor - Tensor with shape `[1, H, W, 3]` and RGB pixels.
 * @returns {{ height: number, width: number, abgr: Buffer }} ABGR bytes for BMP encoding.
 */
function toABGR(tensor) {
  const shape = tensor.getShape();
  if (shape.length !== 4 || shape[0] !== 1) {
    throw new Error(`Expected tensor with shape [1, H, W, 3], got [${shape.join(", ")}]`);
  }
  const [_, height, width, channels] = shape;
  if (channels !== 3) {
    throw new Error(`Expected RGB image tensor, got ${channels} channels.`);
  }

  const rgb = tensor.data instanceof Uint8Array ? tensor.data : Uint8Array.from(tensor.data);
  const abgr = Buffer.allocUnsafe(width * height * 4);

  for (let src = 0, dst = 0; src < rgb.length; src += 3, dst += 4) {
    abgr[dst] = 255;              // A
    abgr[dst + 1] = rgb[src + 2]; // B
    abgr[dst + 2] = rgb[src + 1]; // G
    abgr[dst + 3] = rgb[src];     // R
  }

  return { height, width, abgr };
}

/**
 * Saves an RGB image tensor as a BMP file.
 *
 * @param {string} filePath - Path to the output BMP file.
 * @param {ov.Tensor} tensor - Tensor with shape `[1, H, W, 3]` and RGB pixels.
 */
export async function saveAsBMP(filePath, tensor) {
  const { width, height, abgr } = toABGR(tensor);
  const bmpBuffer = bmp.encode({ data: abgr, width, height }).data;
  await writeFile(filePath, bmpBuffer);
}
