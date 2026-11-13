// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from 'node:path';

import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { Text2SpeechPipeline } from 'openvino-genai-node';
import { readSpeakerEmbedding } from './embedding_utils.js';
import { writeFloatWav } from '../wav_utils.js';

/**
 * Parse CLI arguments, run TTS inference and write the output WAV file.
 * @returns {Promise<void>}
 */
async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      '$0 <model_dir> <text> [device]',
      'Run Text2Speech pipeline on input text and save audio to WAV',
      (yargsBuilder) =>
        yargsBuilder
          .positional('model_dir', {
            type: 'string',
            describe: 'Path to the converted SpeechT5 model directory',
            demandOption: true,
          })
          .positional('text', {
            type: 'string',
            describe: 'Input text to synthesize',
            demandOption: true,
          })
          .positional('device', {
            type: 'string',
            describe: 'Device to run the model on (e.g. CPU, GPU)',
            default: 'CPU',
          })
          .option('speaker_embedding', {
            type: 'string',
            describe:
              'Path to the binary file with a speaker embedding.',
            alias: 's',
          })
          .option('speaker_embedding_file_path', {
            type: 'string',
            describe: 'Path to the binary file with a speaker embedding.',
          })
          .option('language', {
            type: 'string',
            describe: 'Optional language (for example: en-us, en-gb, es, fr-fr, hi, it, pt-br)',
          })
          .option('speed', {
            type: 'number',
            describe: 'Optional speech speed multiplier (Kokoro)',
            default: 1.0,
          })
          .option('output', {
            type: 'string',
            describe: 'Output WAV file path',
            alias: 'o',
            default: 'output_audio.wav',
          }),
    )
    .strict()
    .help()
    .parse();

  const modelDir = argv.model_dir;
  const text = argv.text;
  const device = argv.device;
  const outputPath = argv.output;

  const pipeline = await Text2SpeechPipeline(modelDir, device);

  const generateOptions = {};
  const speakerEmbeddingPath = argv.speaker_embedding_file_path ?? argv.speaker_embedding;
  if (speakerEmbeddingPath) {
    const expectedShape = pipeline.getSpeakerEmbeddingShape();
    generateOptions.speakerEmbedding =
      await readSpeakerEmbedding(speakerEmbeddingPath, expectedShape);
  }

  const generationConfig = {};
  if (argv.language) {
    generationConfig.language = argv.language;
  }
  if (argv.speed !== 1.0) {
    generationConfig.speed = argv.speed;
  }
  if (Object.keys(generationConfig).length > 0) {
    generateOptions.generationConfig = generationConfig;
  }

  const result = await pipeline.generate(text, generateOptions);

  console.assert(result.speeches.length === 1, "Expected only one waveform for the requested input text");

  const sampleRate = result.outputSampleRate;
  await writeFloatWav(outputPath, result.speeches[0].data, sampleRate);

  console.log(`[Info] Text successfully converted to audio file "${outputPath}".`);

  const perfMetrics = result.perfMetrics;
  if (perfMetrics) {
    console.log('\n=== Performance Summary ===');
    console.log(
      'Throughput              :',
      perfMetrics.getThroughput().mean.toFixed(2),
      'samples/sec.',
    );
    console.log(
      'Total Generation Time   :',
      (perfMetrics.getGenerateDuration().mean / 1000.0).toFixed(3),
      'sec.',
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
