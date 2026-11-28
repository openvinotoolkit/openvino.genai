// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { addon as ov } from 'openvino-node';
import { readFile, writeFile } from 'node:fs/promises';

function toNumber(dim) {
  if (typeof dim === 'bigint') {
    if (dim < 0n) {
      throw new Error(`Shape dimension must be non-negative, got ${dim}.`);
    }
    if (dim > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(
        `Shape dimension ${dim} exceeds Number.MAX_SAFE_INTEGER and cannot be represented safely.`,
      );
    }
    return Number(dim);
  }
  if (!Number.isInteger(dim) || dim < 0) {
    throw new Error(`Shape dimension must be a non-negative integer, got ${dim}.`);
  }
  if (!Number.isSafeInteger(dim)) {
    throw new Error(
      `Shape dimension ${dim} exceeds Number.MAX_SAFE_INTEGER and cannot be represented safely.`,
    );
  }
  return dim;
}

function shapeElementCount(shape) {
  return shape.reduce((acc, dim) => acc * toNumber(dim), 1);
}

function toShapeNumbers(shape) {
  return shape.map(toNumber);
}

export async function readSpeakerEmbedding(filePath, expectedShape = [1, 512]) {
    const buf = await readFile(filePath);
    const shape = toShapeNumbers(expectedShape);
    const expectedLength = shapeElementCount(shape);
    const expectedBytes = expectedLength * Float32Array.BYTES_PER_ELEMENT;

    if (buf.byteLength !== expectedBytes) {
      throw new Error(
        `Speaker embedding file must contain exactly ${expectedLength} float32 `
        + `values (${expectedBytes} bytes), got ${buf.byteLength} bytes`,
      );
    }

    const floats = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      expectedLength,
    );

    return new ov.Tensor('f32', shape, floats);
}

export async function writeSpeakerEmbedding(filePath, tensor) {
  const data = tensor.data;
  const bytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  await writeFile(filePath, bytes);
}

export async function writeReferenceCodecIds(filePath, tensor) {
  const shape = tensor.getShape().map(toNumber);
  const rank = shape.length;
  const data = tensor.data;
  const payload = BigInt64Array.from(data, (v) => BigInt(v));

  const header = Buffer.alloc((1 + rank) * 8);
  let offset = 0;
  header.writeBigInt64LE(BigInt(rank), offset);
  offset += 8;
  for (const dim of shape) {
    header.writeBigInt64LE(BigInt(dim), offset);
    offset += 8;
  }

  const payloadBuffer = Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength);
  await writeFile(filePath, Buffer.concat([header, payloadBuffer]));
}

export async function readReferenceCodecIds(filePath) {
  const buf = await readFile(filePath);
  if (buf.byteLength < 8) {
    throw new Error(`Invalid reference codec ids file: ${filePath}`);
  }

  let offset = 0;
  const rank = Number(buf.readBigInt64LE(offset));
  offset += 8;
  if (rank <= 0 || rank > 8) {
    throw new Error(`Invalid reference codec ids rank in file: ${filePath}`);
  }

  const shape = [];
  for (let i = 0; i < rank; i += 1) {
    if (offset + 8 > buf.byteLength) {
      throw new Error(`Invalid reference codec ids header in file: ${filePath}`);
    }
    shape.push(Number(buf.readBigInt64LE(offset)));
    offset += 8;
  }

  const payloadBytes = buf.byteLength - offset;
  if (payloadBytes % 8 !== 0) {
    throw new Error(`Invalid reference codec ids payload size in file: ${filePath}`);
  }

  const count = payloadBytes / 8;
  const values = new BigInt64Array(count);
  for (let i = 0; i < count; i += 1) {
    values[i] = buf.readBigInt64LE(offset + i * 8);
  }

  return new ov.Tensor('i64', shape, values);
}
