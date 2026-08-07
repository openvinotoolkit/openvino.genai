// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from 'node:path';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { ASRPipeline } from 'openvino-genai-node';
import { readAudio } from './wav_utils.js';

/**
 * Parse CLI arguments, run ASR inference and print transcription output.
 * @returns {Promise<void>}
 */
async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      '$0 <model_dir> <audio_file> [device]',
      'Run automatic speech recognition on an audio file',
      (yargsBuilder) =>
        yargsBuilder
          .positional('model_dir', {
            type: 'string',
            describe: 'Path to the converted ASR model directory',
            demandOption: true,
          })
          .positional('audio_file', {
            type: 'string',
            describe: 'Path to the WAV audio file',
            demandOption: true,
          })
          .positional('device', {
            type: 'string',
            describe: 'Device to run the model on (e.g. CPU, GPU)',
            default: 'CPU',
          }),
    )
    .strict()
    .help()
    .parse();

  const modelDir = argv.model_dir;
  const wavFilePath = argv.audio_file;
  const device = argv.device;

  const properties = {};
  if (device === 'NPU' || device.startsWith('GPU')) {
    // Cache compiled models on disk for GPU and NPU to save time on the next run.
    properties['CACHE_DIR'] = 'asr_cache';
  }
  // Word timestamps supported by Whisper models only.
  // Must be passed to the ASRPipeline constructor as a property.
  properties.word_timestamps = true;

  const pipeline = await ASRPipeline(modelDir, device, properties);

  // If the language is known in advance it can be specified in the generation config.
  // In the form of "<|en|>" for Whisper models. Supported by multilingual models only.
  // In the form of "English" for Qwen3-ASR models.
  // Whisper model parameters (task, return_timestamps, word_timestamps) are ignored for Qwen3-ASR models.
  const generationConfig = {
    language: '<|en|>',
    task: 'transcribe',
    return_timestamps: true,
    word_timestamps: true,
  };

  // Pipeline expects normalized audio with a sample rate of 16 kHz.
  const audioTensor = await readAudio(wavFilePath);
  const result = await pipeline.generate(audioTensor, { generationConfig });

  console.log(result.texts?.[0] ?? '');

  if (result.chunks?.[0]?.length) {
    for (const chunk of result.chunks[0]) {
      console.log(`timestamps: [${chunk.startTs.toFixed(2)}, ${chunk.endTs.toFixed(2)}] text: ${chunk.text}`);
    }
  }

  if (result.words?.[0]?.length) {
    for (const word of result.words[0]) {
      console.log(`[${word.startTs.toFixed(2)}, ${word.endTs.toFixed(2)}]: ${word.text}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
