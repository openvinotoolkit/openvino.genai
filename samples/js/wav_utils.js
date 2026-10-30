// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readFile, writeFile } from 'node:fs/promises';
import { encode } from 'node-wav';

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
 * Read WAV file and convert to 16kHz mono Float32Array for speech recognition pipelines.
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

/**
 * Save an audio tensor as a 32-bit float WAV file.
 * If the tensor holds several tracks, each one is saved to its own file with a "_b<index>" suffix.
 * @param {string} filename - Path to the output WAV file.
 * @param {Tensor} audioTensor - f32 tensor with shape [num_tracks, num_channels, num_samples].
 * @param {number} sampleRate - Sample rate of the audio in Hz.
 * @returns {Promise<void>}
 */
export async function saveAudio(filename, audioTensor, sampleRate) {
  const [batchSize, numChannels, numSamples] = audioTensor.getShape();
  const audioData = audioTensor.data;

  for (let b = 0; b < batchSize; b++) {
    let outputPath = filename;
    if (batchSize > 1) {
      const dot = filename.lastIndexOf('.');
      const base = dot === -1 ? filename : filename.slice(0, dot);
      const ext = dot === -1 ? 'wav' : filename.slice(dot + 1);
      outputPath = `${base}_b${b}.${ext}`;
    }

    // node-wav takes one Float32Array per channel
    const channelData = [];
    for (let c = 0; c < numChannels; c++) {
      const offset = (b * numChannels + c) * numSamples;
      channelData.push(audioData.subarray(offset, offset + numSamples));
    }

    await writeFile(outputPath, encode(channelData, { sampleRate, float: true }));
    console.log(`Wrote ${outputPath} (${numSamples} samples, ${numChannels} channels @ ${sampleRate} Hz)`);
  }
}
