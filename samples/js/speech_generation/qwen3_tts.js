// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from 'node:path';

import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import { Text2SpeechPipeline } from 'openvino-genai-node';
import { readWavAsTensor, writeFloatWav } from '../wav_utils.js';
import {
  readReferenceCodecIds,
  readSpeakerEmbedding,
  writeReferenceCodecIds,
  writeSpeakerEmbedding,
} from './embedding_utils.js';

function getPipelinePropertiesForDevice(device) {
  const upper = device.toUpperCase();
  if (upper.includes('GPU') || upper.includes('NPU')) {
    return { CACHE_DIR: 'qwen3_tts_cache_dir' };
  }
  return {};
}

async function writeAudioAndPerf(result, outputPath) {
  console.assert(result.speeches.length === 1, 'Expected only one waveform for the requested input text');

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

async function runBase(pipeline, argv) {
  if (!argv.ref_audio_wav_path && !argv.speaker_embedding_file_path) {
    throw new Error('Qwen3-TTS Base requires --ref_audio_wav_path or --speaker_embedding_file_path.');
  }

  if (argv.ref_codec_ids_file_path && !argv.ref_text) {
    throw new Error('--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).');
  }

  const generateOptions = { generationConfig: {} };

  if (argv.language) {
    generateOptions.generationConfig.language = argv.language;
  }
  if (argv.ref_audio_wav_path) {
    generateOptions.generationConfig.ref_audio = await readWavAsTensor(argv.ref_audio_wav_path, 24000);
  }
  if (argv.ref_text) {
    generateOptions.generationConfig.ref_text = argv.ref_text;
  }
  if (argv.ref_codec_ids_file_path) {
    generateOptions.generationConfig.ref_codec_ids =
      await readReferenceCodecIds(argv.ref_codec_ids_file_path);
  }
  if (typeof argv.max_new_tokens === 'number') {
    generateOptions.generationConfig.max_new_tokens = argv.max_new_tokens;
  }
  if (typeof argv.seed === 'number') {
    generateOptions.generationConfig.rng_seed = argv.seed;
  }

  if (argv.speaker_embedding_file_path) {
    generateOptions.speakerEmbedding = await readSpeakerEmbedding(
      argv.speaker_embedding_file_path,
      pipeline.getSpeakerEmbeddingShape(),
    );
  }

  const mode = argv.ref_text ? 'ICL' : 'x-vector';
  console.log(`[Info] Qwen3-TTS Base voice clone (${mode} mode).`);

  const result = await pipeline.generate(argv.text, generateOptions);
  await writeAudioAndPerf(result, argv.output_wav_path);

  if (argv.save_speaker_embedding_file_path) {
    if (!result.speakerEmbedding) {
      throw new Error('No speaker embedding was produced to save. Provide --ref_audio_wav_path.');
    }
    await writeSpeakerEmbedding(argv.save_speaker_embedding_file_path, result.speakerEmbedding);
    console.log(`[Info] Saved speaker embedding to "${argv.save_speaker_embedding_file_path}".`);
  }
  if (argv.save_ref_codec_ids_file_path) {
    if (!result.refCodecIds) {
      throw new Error(
        'No reference codes were produced to save. ICL mode (--ref_text) with --ref_audio_wav_path is required.',
      );
    }
    await writeReferenceCodecIds(argv.save_ref_codec_ids_file_path, result.refCodecIds);
    console.log(`[Info] Saved reference codes to "${argv.save_ref_codec_ids_file_path}".`);
  }
}

async function runCustomVoice(pipeline, argv) {
  if (!argv.speaker) {
    throw new Error('Qwen3-TTS CustomVoice requires --speaker <NAME>.');
  }

  const generationConfig = { speaker: argv.speaker };
  if (argv.language) {
    generationConfig.language = argv.language;
  }
  if (argv.instruct) {
    generationConfig.instruct = argv.instruct;
  }
  if (typeof argv.max_new_tokens === 'number') {
    generationConfig.max_new_tokens = argv.max_new_tokens;
  }
  if (typeof argv.seed === 'number') {
    generationConfig.rng_seed = argv.seed;
  }

  const result = await pipeline.generate(argv.text, { generationConfig });
  await writeAudioAndPerf(result, argv.output_wav_path);
}

async function runVoiceDesign(pipeline, argv) {
  if (!argv.instruct) {
    throw new Error('Qwen3-TTS VoiceDesign requires --instruct "<VOICE_DESCRIPTION>".');
  }

  const generationConfig = { instruct: argv.instruct };
  if (argv.language) {
    generationConfig.language = argv.language;
  }
  if (typeof argv.max_new_tokens === 'number') {
    generationConfig.max_new_tokens = argv.max_new_tokens;
  }
  if (typeof argv.seed === 'number') {
    generationConfig.rng_seed = argv.seed;
  }

  const result = await pipeline.generate(argv.text, { generationConfig });
  await writeAudioAndPerf(result, argv.output_wav_path);
}

async function main() {
  const argv = yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      '$0 <variant> <model_dir> <text>',
      'Run Qwen3-TTS sample (base, customvoice, or voice-design).',
      (yargsBuilder) =>
        yargsBuilder
          .positional('variant', {
            type: 'string',
            choices: ['base', 'customvoice', 'voice-design'],
            demandOption: true,
          })
          .positional('model_dir', {
            type: 'string',
            describe: 'Path to Qwen3-TTS model directory',
            demandOption: true,
          })
          .positional('text', {
            type: 'string',
            describe: 'Input text to synthesize',
            demandOption: true,
          })
          .option('ref_audio_wav_path', {
            type: 'string',
            describe: 'Reference audio WAV path (Base variant only, 24000 Hz)',
          })
          .option('speaker_embedding_file_path', {
            type: 'string',
            describe: 'Pre-saved speaker embedding .bin (Base variant only)',
          })
          .option('ref_text', {
            type: 'string',
            describe: 'Reference transcript for ICL mode (Base variant)',
          })
          .option('ref_codec_ids_file_path', {
            type: 'string',
            describe: 'Pre-saved reference codec ids .bin (Base variant)',
          })
          .option('save_speaker_embedding_file_path', {
            type: 'string',
            describe: 'Save extracted speaker embedding to .bin (Base variant)',
          })
          .option('save_ref_codec_ids_file_path', {
            type: 'string',
            describe: 'Save extracted reference codec ids to .bin (Base variant)',
          })
          .option('speaker', {
            type: 'string',
            describe: 'Predefined speaker name (CustomVoice variant)',
          })
          .option('language', {
            type: 'string',
            describe: 'Optional language (for example: english)',
          })
          .option('instruct', {
            type: 'string',
            describe: 'Style instruction or voice description',
          })
          .option('device', {
            type: 'string',
            describe: 'Device to run the model on',
            default: 'CPU',
          })
          .option('max_new_tokens', {
            type: 'number',
            describe: 'Optional max new tokens override',
          })
          .option('seed', {
            type: 'number',
            describe: 'Optional RNG seed override',
          })
          .option('output_wav_path', {
            type: 'string',
            describe: 'Output WAV path',
            default: 'output_audio.wav',
          }),
    )
    .strict()
    .help()
    .parse();

  const pipeline = await Text2SpeechPipeline(
    argv.model_dir,
    argv.device,
    getPipelinePropertiesForDevice(argv.device),
  );

  if (argv.variant === 'base') {
    await runBase(pipeline, argv);
  } else if (argv.variant === 'customvoice') {
    await runCustomVoice(pipeline, argv);
  } else {
    await runVoiceDesign(pipeline, argv);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
