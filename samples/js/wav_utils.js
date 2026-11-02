// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { addon as ov } from 'openvino-node';
import { readFile, writeFile } from 'node:fs/promises';
import { decode, encode } from 'node-wav';

function decodeFloat32WavWithDataView(wavBuffer) {
  const bytes = Uint8Array.from(wavBuffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.byteLength < 44) {
    throw new Error('WAV file is too small to contain a valid header.');
  }

  const riff = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  const wave = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    throw new Error('Unsupported WAV container. Expected RIFF/WAVE.');
  }

  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = -1;

  while (offset + 8 <= bytes.byteLength) {
    const chunkId = String.fromCharCode(
      bytes[offset],
      bytes[offset + 1],
      bytes[offset + 2],
      bytes[offset + 3],
    );
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;

    if (chunkDataOffset + chunkSize > bytes.byteLength) {
      throw new Error(
        `Malformed WAV: chunk "${chunkId}" at offset ${offset} declares ${chunkSize} bytes, `
        + `which exceeds file size ${bytes.byteLength}.`,
      );
    }

    if (chunkId === 'fmt ' && chunkSize >= 16) {
      fmt = {
        audioFormat: view.getUint16(chunkDataOffset, true),
        numChannels: view.getUint16(chunkDataOffset + 2, true),
        sampleRate: view.getUint32(chunkDataOffset + 4, true),
        blockAlign: view.getUint16(chunkDataOffset + 12, true),
        bitsPerSample: view.getUint16(chunkDataOffset + 14, true),
      };
    } else if (chunkId === 'data') {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
    }

    // WAV chunks are word-aligned.
    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (!fmt || dataOffset < 0 || dataSize < 0) {
    throw new Error('WAV file must contain both fmt and data chunks.');
  }
  if (fmt.audioFormat !== 3 || fmt.bitsPerSample !== 32) {
    throw new Error(
      `Fallback decoder supports only float32 WAV (format=3, bits=32), got format=${fmt.audioFormat}, bits=${fmt.bitsPerSample}.`,
    );
  }
  if (fmt.numChannels <= 0) {
    throw new Error('WAV has invalid channel count.');
  }
  if (dataSize % fmt.blockAlign !== 0) {
    throw new Error('WAV data chunk size is not aligned with block alignment.');
  }

  const samplesPerChannel = dataSize / fmt.blockAlign;
  const channelData = Array.from({ length: fmt.numChannels }, () => new Float32Array(samplesPerChannel));

  for (let i = 0; i < samplesPerChannel; i += 1) {
    for (let ch = 0; ch < fmt.numChannels; ch += 1) {
      const sampleOffset = dataOffset + (i * fmt.numChannels + ch) * 4;
      channelData[ch][i] = view.getFloat32(sampleOffset, true);
    }
  }

  return {
    sampleRate: fmt.sampleRate,
    channelData,
  };
}

function decodeWavRobust(wavBuffer) {
  try {
    return decode(wavBuffer);
  } catch (err) {
    if (
      err instanceof RangeError
      && typeof err.message === 'string'
      && err.message.includes('start offset of Float32Array should be a multiple of 4')
    ) {
      return decodeFloat32WavWithDataView(wavBuffer);
    }
    throw err;
  }
}

/**
 * Read WAV file and convert to mono Float32Array.
 * @param {string} audioPath
 * @param {number|undefined} expectedSampleRate
 * @returns {Promise<Float32Array>}
 */
export async function readAudio(audioPath, expectedSampleRate = 16000) {
  const wavBuffer = await readFile(audioPath);

  if (wavBuffer.length === 0) {
    throw new Error('Audio file is empty.');
  }

  const decoded = decodeWavRobust(wavBuffer);
  if (expectedSampleRate !== undefined && decoded.sampleRate !== expectedSampleRate) {
    throw new Error(`WAV file must be ${expectedSampleRate} Hz, but got ${decoded.sampleRate}.`);
  }

  const channels = decoded.channelData;
  if (!channels || channels.length === 0) {
    throw new Error('WAV file contains no channels.');
  }

  if (channels.length === 1) {
    return channels[0];
  }

  const length = channels[0].length;
  const mono = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    let sum = 0;
    for (const channel of channels) {
      sum += channel[i];
    }
    mono[i] = sum / channels.length;
  }
  return mono;
}

/**
 * Read WAV file and wrap mono float samples as an f32 OpenVINO tensor.
 * @param {string} audioPath
 * @param {number|undefined} expectedSampleRate
 * @returns {Promise<import('openvino-node').Tensor>}
 */
export async function readWavAsTensor(audioPath, expectedSampleRate = 16000) {
  const mono = await readAudio(audioPath, expectedSampleRate);
  return new ov.Tensor('f32', [mono.length], mono);
}

/**
 * Write mono samples as IEEE float WAV (format=3, 32-bit float).
 * @param {string} audioPath
 * @param {Float32Array|number[]} monoSamples
 * @param {number} sampleRate
 * @returns {Promise<void>}
 */
export async function writeFloatWav(audioPath, monoSamples, sampleRate) {
  const samples = monoSamples instanceof Float32Array ? monoSamples : Float32Array.from(monoSamples);
  const wavData = encode([samples], { sampleRate, float: true, bitDepth: 32 });
  await writeFile(audioPath, wavData);
}
