// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises';

function parseWavPcm16Mono(buffer) {
  if (buffer.length < 44) {
    throw new Error('Invalid WAV payload: file is too small.');
  }

  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Invalid WAV payload: RIFF/WAVE header is missing.');
  }

  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = 12;

  let audioFormat;
  let channels;
  let sampleRate;
  let bitsPerSample;
  let dataOffset;
  let dataSize;

  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;

    if (chunkDataOffset + chunkSize > buffer.length) {
      throw new Error('Invalid WAV payload: malformed chunk size.');
    }

    if (chunkId === 'fmt ') {
      if (chunkSize < 16) {
        throw new Error('Invalid WAV payload: fmt chunk is too small.');
      }
      audioFormat = view.getUint16(chunkDataOffset, true);
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
    } else if (chunkId === 'data') {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (audioFormat !== 1) {
    throw new Error('Unsupported WAV format: only PCM is supported.');
  }

  if (channels !== 1 && channels !== 2) {
    throw new Error('WAV file must be mono or stereo.');
  }

  if (sampleRate !== 16000) {
    throw new Error(`WAV file must be 16 kHz, but got ${sampleRate}.`);
  }

  if (bitsPerSample !== 16) {
    throw new Error(`Unsupported WAV bit depth: ${bitsPerSample}. Only 16-bit PCM is supported.`);
  }

  if (dataOffset === undefined || dataSize === undefined) {
    throw new Error('Invalid WAV payload: missing data chunk.');
  }

  const bytesPerFrame = channels * 2;
  const frameCount = Math.floor(dataSize / bytesPerFrame);
  const mono = new Float32Array(frameCount);

  for (let index = 0; index < frameCount; index++) {
    const frameOffset = dataOffset + index * bytesPerFrame;
    if (channels === 1) {
      const sample = view.getInt16(frameOffset, true);
      mono[index] = sample / 32768.0;
    } else {
      const left = view.getInt16(frameOffset, true);
      const right = view.getInt16(frameOffset + 2, true);
      mono[index] = (left + right) / 65536.0;
    }
  }

  return mono;
}

/**
 * Read WAV file and convert to 16kHz mono Float32Array for Whisper pipeline.
 * @param {string} audioPath
 * @returns {Promise<Float32Array>}
 */
export async function readAudio(audioPath) {
  const wavBuffer = await readFile(audioPath);

  if (wavBuffer.length === 0) {
    throw new Error('Audio file is empty.');
  }

  return parseWavPcm16Mono(wavBuffer);
}
