// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { addon as ov } from 'openvino-node';
import { readFile } from 'node:fs/promises';

export async function readSpeakerEmbedding(filePath) {
    const buf = await readFile(filePath);
    const expectedLength = 512;
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

    return new ov.Tensor('f32', [1, expectedLength], floats);
}
