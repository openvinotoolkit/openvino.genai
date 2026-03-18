// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';

import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { decode, encode } from 'node-wav';
import { addon as ov } from 'openvino-node';
import { Text2SpeechPipeline } from 'openvino-genai-node';

async function readSpeakerEmbedding(filePath) {
  const embeddingWav = await readFile(filePath);
  const decodedEmbedding = decode(embeddingWav);
  const maxEmbeddingLength = 512;
  return new ov.Tensor('f32', [1, maxEmbeddingLength], decodedEmbedding.channelData[0].slice(0, maxEmbeddingLength));
}

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
              'Path to the binary file with a speaker embedding '
              + '(512 float32 values)',
            alias: 's',
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
  if (argv.speaker_embedding) {
    generateOptions.speakerEmbedding =
      await readSpeakerEmbedding(argv.speaker_embedding);
  }

  const result = await pipeline.generate(text, generateOptions);

  const sampleRate = 16000;
  const wavData = encode([result.speeches[0].data], { sampleRate });
  await writeFile(outputPath, wavData);

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
