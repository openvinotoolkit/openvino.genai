// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises';
import wav from 'node-wav';

/**
 * Convert decoded multi-channel PCM data to mono by averaging channels.
 * @param {Float32Array[]} channelData
 * @returns {Float32Array}
 */
function toMono(channelData) {
  if (!Array.isArray(channelData) || channelData.length === 0) {
    throw new Error('Invalid WAV payload: no channel data.');
  }

  if (channelData.length === 1) {
    return channelData[0];
  }

  const minLength = Math.min(...channelData.map((channel) => channel.length));
  if (!Number.isFinite(minLength) || minLength <= 0) {
    throw new Error('Invalid WAV payload: channel data is empty.');
  }

  const mono = new Float32Array(minLength);
  for (let index = 0; index < minLength; index++) {
    let sum = 0;
    for (const channel of channelData) {
      sum += channel[index];
    }
    mono[index] = sum / channelData.length;
  }

  return mono;
}

/**
 * Resample PCM data with linear interpolation.
 * @param {Float32Array} samples
 * @param {number} inputRate
 * @param {number} outputRate
 * @returns {Float32Array}
 */
function resampleLinear(samples, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return samples;
  }

  if (!Number.isFinite(inputRate) || inputRate <= 0) {
    throw new Error(`Invalid WAV sample rate: ${inputRate}`);
  }

  const outputLength = Math.max(1, Math.round((samples.length * outputRate) / inputRate));
  const resampled = new Float32Array(outputLength);
  const ratio = inputRate / outputRate;

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex++) {
    const inputPosition = outputIndex * ratio;
    const leftIndex = Math.floor(inputPosition);
    const rightIndex = Math.min(leftIndex + 1, samples.length - 1);
    const weight = inputPosition - leftIndex;
    const leftValue = samples[leftIndex] ?? 0;
    const rightValue = samples[rightIndex] ?? leftValue;
    resampled[outputIndex] = leftValue + (rightValue - leftValue) * weight;
  }

  return resampled;
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

  const decoded = wav.decode(wavBuffer);
  const sampleRate = Number(decoded?.sampleRate);

  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error('Unable to read WAV sample rate.');
  }

  const mono = toMono(decoded.channelData);
  if (mono.length === 0) {
    throw new Error('Decoded audio contains 0 samples.');
  }

  const whisperSampleRate = 16_000;
  return resampleLinear(mono, sampleRate, whisperSampleRate);
}
