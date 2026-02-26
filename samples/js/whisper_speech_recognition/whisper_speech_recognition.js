// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from 'node:path';
import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { WhisperPipeline } from 'openvino-genai-node';
import { readAudio } from '../ffmpeg_utils.js';

function getConfigForCache() {
  const config = { CACHE_DIR: 'whisper_cache' };
  return config;
}

async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      '$0 <model_dir> <audio_file> [device]',
      'Run Whisper speech recognition on an audio file',
      (yargsBuilder) =>
        yargsBuilder
          .positional('model_dir', {
            type: 'string',
            describe: 'Path to the converted Whisper model directory',
            demandOption: true,
          })
          .positional('audio_file', {
            type: 'string',
            describe: 'Path to the audio file (WAV, MP3, M4A, etc.; decoded via ffmpeg to 16 kHz mono)',
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

  let properties = {};
  if (device === 'NPU' || device.startsWith('GPU')) {
    properties = getConfigForCache();
  }
  // Word timestamps require word_timestamps in the pipeline constructor
  properties.word_timestamps = true;

  const pipeline = await WhisperPipeline(modelDir, device, properties);

  // Pass only the options to override; avoid spreading full getGenerationConfig()
  // (it can contain values that do not round-trip correctly, e.g. max_new_tokens).
  const generationConfig = {
    language: '<|en|>',
    task: 'transcribe',
    return_timestamps: true,
    word_timestamps: true,
  };

  const audioTensor = await readAudio(wavFilePath);
  const result = await pipeline.generate(audioTensor, { generationConfig });

  console.log(result.texts?.[0] ?? '');

  if (result.chunks?.length) {
    for (const chunk of result.chunks) {
      console.log(`timestamps: [${chunk.startTs.toFixed(2)}, ${chunk.endTs.toFixed(2)}] text: ${chunk.text}`);
    }
  }

  if (result.words?.length) {
    for (const word of result.words) {
      console.log(`[${word.startTs.toFixed(2)}, ${word.endTs.toFixed(2)}]: ${word.word}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
