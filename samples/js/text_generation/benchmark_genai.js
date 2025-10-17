// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LLMPipeline } from "openvino-genai-node";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { readFileSync } from "fs";

main();

async function main() {
  const argv = yargs(hideBin(process.argv))
    .option("model", {
      alias: "m",
      type: "string",
      demandOption: true,
      describe: "Path to model and tokenizers base directory.",
    })
    .option("prompt", {
      alias: "p",
      type: "string",
      describe:
        "The prompt to generate text. If without `-p` and `--pf`, the default prompt is `The Sky is blue because`.",
    })
    .option("prompt_file", {
      alias: "pf",
      type: "string",
      describe: "Read prompt from file.",
    })
    .option("num_warmup", {
      alias: "nw",
      type: "number",
      default: 1,
      describe: "Number of warmup iterations.",
    })
    .option("num_iter", {
      alias: "n",
      type: "number",
      default: 2,
      describe: "Number of iterations.",
    })
    .option("max_new_tokens", {
      alias: "mt",
      type: "number",
      default: 20,
      describe: "Maximal number of new tokens.",
    })
    .option("device", {
      alias: "d",
      type: "string",
      default: "CPU",
      describe: "Device.",
    })
    .parse();

  let prompt;
  if (argv.prompt !== undefined && argv.prompt_file !== undefined) {
    console.error(`Cannot specify both --prompt and --prompt_file options simultaneously!`);
    process.exit(1);
  } else {
    if (argv.prompt_file !== undefined) {
      prompt = [readFileSync(argv.prompt_file, "utf-8")];
    } else {
      prompt = argv.prompt === undefined ? ["The Sky is blue because"] : [argv.prompt];
    }
  }
  if (prompt.length === 0 || prompt[0].trim() === "") {
    throw new Error("Prompt is empty!");
  }

  const modelsPath = argv.model;
  const { device } = argv;
  const numWarmup = argv.num_warmup;
  const numIter = argv.num_iter;

  const config = {
    max_new_tokens: argv.max_new_tokens,
    apply_chat_template: false,
    return_decoded_results: true,
  };

  let pipe;
  if (device === "NPU") {
    pipe = await LLMPipeline(modelsPath, device);
  } else {
    const schedulerConfig = {
      enable_prefix_caching: false,
      max_num_batched_tokens: Number.MAX_SAFE_INTEGER,
    };
    pipe = await LLMPipeline(modelsPath, device, { schedulerConfig: schedulerConfig });
  }

  for (let i = 0; i < numWarmup; i++) {
    await pipe.generate(prompt, config);
  }

  let res = await pipe.generate(prompt, config);
  let { perfMetrics } = res;
  for (let i = 0; i < numIter - 1; i++) {
    res = await pipe.generate(prompt, config);
    perfMetrics.add(res.perfMetrics);
  }

  console.log(`Output token size: ${perfMetrics.getNumGeneratedTokens()}`);
  console.log(`Load time: ${perfMetrics.getLoadTime()} ms`);
  console.log(`Generate time: ${perfMetrics.getGenerateDuration().mean} ± ${perfMetrics.getGenerateDuration().std} ms`);
  console.log(`Tokenization time: ${perfMetrics.getTokenizationDuration().mean} ± ${perfMetrics.getTokenizationDuration().std} ms`);
  console.log(`Detokenization time: ${perfMetrics.getDetokenizationDuration().mean} ± ${perfMetrics.getDetokenizationDuration().std} ms`);
  console.log(`TTFT: ${perfMetrics.getTTFT().mean} ± ${perfMetrics.getTTFT().std} ms`);
  console.log(`TPOT: ${perfMetrics.getTPOT().mean} ± ${perfMetrics.getTPOT().std} ms`);
  console.log(`Throughput : ${perfMetrics.getThroughput().mean} ± ${perfMetrics.getThroughput().std} tokens/s`);
}
