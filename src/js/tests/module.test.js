import { LLMPipeline } from "../dist/index.js";

import assert from "node:assert/strict";
import { describe, it, before, after } from "node:test";
import { models } from "./models.js";
import { hrtime } from "node:process";
import os from "node:os";

const MODEL_PATH = process.env.MODEL_PATH || `./tests/models/${models.LLM.split("/")[1]}`;

describe("LLMPipeline construction", async () => {
  await it("test LLMPipeline(modelPath)", async () => {
    const pipe = await LLMPipeline(MODEL_PATH);
    assert.strictEqual(typeof pipe, "object");
  });

  await it("test error LLMPipeline(modelPath, LLMPipelineProperties)", async () => {
    // Test for JS. In TS it won't be possible to call LLMPipeline with wrong types.
    await assert.rejects(async () => await LLMPipeline(MODEL_PATH, {}), {
      name: "Error",
      message:
        "The second argument must be a device string. If you want to pass LLMPipelineProperties, please use the third argument.",
    });
  });

  await it("test SchedulerConfig", async (t) => {
    if (os.platform() === "darwin") {
      t.skip("only support x64 platform or ARM with SVE support");
      return;
    }
    const schedulerConfig = {
      max_num_batched_tokens: 32,
    };
    assert.ok(await LLMPipeline(MODEL_PATH, "CPU", { schedulerConfig: schedulerConfig }));
  });
});

describe("LLMPipeline methods", async () => {
  let pipeline = null;

  await before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await pipeline.startChat();
  });

  await after(async () => {
    await pipeline.finishChat();
  });

  await it("should generate non empty string", async () => {
    const result = await pipeline.generate(
      "Type something in English",
      { temperature: "0", max_new_tokens: "4" },
      () => {},
    );

    assert.ok(result.length > 0);
    assert.strictEqual(typeof result, "string");
  });

  it("should include tokenizer", async () => {
    const tokenizer = pipeline.getTokenizer();
    assert.strictEqual(typeof tokenizer, "object");
  });
});

describe("corner cases", async () => {
  it("should throw an error if pipeline is already initialized", async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await assert.rejects(async () => await pipeline.init(), {
      name: "Error",
      message: "LLMPipeline is already initialized",
    });
  });

  it("should throw an error if chat is already started", async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await pipeline.startChat();

    await assert.rejects(() => pipeline.startChat(), {
      name: "Error",
      message: "Chat is already started",
    });
  });

  it("should throw an error if chat is not started", async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await assert.rejects(() => pipeline.finishChat(), {
      name: "Error",
      message: "Chat is not started",
    });
  });
});

describe("generation parameters validation", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await pipeline.startChat();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it("should throw an error if temperature is not a number", async () => {
    await assert.rejects(async () => await pipeline.generate(), {
      name: "Error",
      message: "Prompt must be a string or string[]",
    });
  });

  it("should throw an error if generationCallback is not a function", async () => {
    const pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await pipeline.startChat();

    await assert.rejects(async () => await pipeline.generate("prompt", {}, false), {
      name: "Error",
      message: "Callback must be a function",
    });
  });

  it("should throw an error if options specified but not an object", async () => {
    await assert.rejects(async () => await pipeline.generate("prompt", "options", () => {}), {
      name: "Error",
      message: "Options must be an object",
    });
  });

  it("should perform generation with default options", async () => {
    try {
      await pipeline.generate("prompt", { max_new_tokens: 1 });
    } catch (error) {
      assert.fail(error);
    }

    assert.ok(true);
  });

  it("should return a string as generation result", async () => {
    const reply = await pipeline.generate("prompt", { max_new_tokens: 1 });

    assert.strictEqual(typeof reply, "string");
  });

  it("should call generationCallback with string chunk", async () => {
    await pipeline.generate("prompt", { max_new_tokens: 1 }, (chunk) => {
      assert.strictEqual(typeof chunk, "string");
    });
  });

  it("should convert Set", async () => {
    const generationConfig = {
      max_new_tokens: 100,
      stop_strings: new Set(["1", "2", "3", "4", "5"]),
      include_stop_str_in_output: true,
    };
    const result = await pipeline.generate("continue: 1 2 3", generationConfig);
    assert.strictEqual(typeof result, "string");
  });
});

describe("LLMPipeline.generate()", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");
    await pipeline.startChat();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it("generate(prompt, config) return_decoded_results", async () => {
    const config = {
      max_new_tokens: 5,
      return_decoded_results: true,
    };
    const reply = await pipeline.generate("prompt", config);
    assert.strictEqual(typeof reply, "object");
    assert.ok(Array.isArray(reply.texts));
    assert.ok(reply.texts.every((text) => typeof text === "string"));
    assert.ok(reply.perfMetrics !== undefined);

    const configStr = {
      max_new_tokens: 5,
      return_decoded_results: false,
    };
    const replyStr = await pipeline.generate("prompt", configStr);
    assert.strictEqual(typeof replyStr, "string");
    assert.strictEqual(replyStr, reply.toString());
  });

  it("DecodedResults.perfMetrics", async (t) => {
    if (os.platform() === "darwin") {
      t.skip("Skipping perfMetrics test on macOS. Ticket - 173286");
      return;
    }

    const config = {
      max_new_tokens: 20,
      return_decoded_results: true,
    };
    const prompt = "The Sky is blue because";
    const start = hrtime.bigint();
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");
    await pipeline.startChat();
    const res = await pipeline.generate(prompt, config);
    const totalTime = Number(hrtime.bigint() - start) / 1e6;

    const { perfMetrics } = res;
    const loadTime = perfMetrics.getLoadTime();
    assert.ok(loadTime >= 0 && loadTime <= totalTime);

    const numGeneratedTokens = perfMetrics.getNumGeneratedTokens();
    assert.ok(numGeneratedTokens > 0);
    assert.ok(numGeneratedTokens <= config.max_new_tokens);

    const numInputTokens = perfMetrics.getNumInputTokens();
    assert.ok(numInputTokens > 0 && typeof numInputTokens === "number");

    const ttft = perfMetrics.getTTFT();
    assert.ok(ttft.mean >= 0 && ttft.mean < 1000.0);
    assert.ok(typeof ttft.std === "number");

    const tpot = perfMetrics.getTPOT();
    assert.ok(tpot.mean >= 0 && tpot.mean < 1000.0);
    assert.ok(typeof tpot.std === "number");

    const throughput = perfMetrics.getThroughput();
    assert.ok(throughput.mean >= 0 && throughput.mean < 20000.0);
    assert.ok(typeof throughput.std === "number");

    const inferenceDuration = perfMetrics.getInferenceDuration();
    assert.ok(inferenceDuration.mean >= 0 && loadTime + inferenceDuration.mean < totalTime);
    assert.strictEqual(inferenceDuration.std, 0);

    const generateDuration = perfMetrics.getGenerateDuration();
    assert.ok(generateDuration.mean >= 0 && loadTime + generateDuration.mean < totalTime);
    assert.strictEqual(generateDuration.std, 0);

    const tokenizationDuration = perfMetrics.getTokenizationDuration();
    assert.ok(tokenizationDuration.mean >= 0 && tokenizationDuration.mean < generateDuration.mean);
    assert.strictEqual(tokenizationDuration.std, 0);

    const detokenizationDuration = perfMetrics.getDetokenizationDuration();
    assert.ok(
      detokenizationDuration.mean >= 0 && detokenizationDuration.mean < generateDuration.mean,
    );
    assert.strictEqual(detokenizationDuration.std, 0);

    assert.ok(typeof perfMetrics.getGrammarCompilerInitTimes() === "object");
    const grammarCompileTime = perfMetrics.getGrammarCompileTime();
    assert.ok(typeof grammarCompileTime.mean === "number");
    assert.ok(typeof grammarCompileTime.std === "number");
    assert.ok(typeof grammarCompileTime.min === "number");
    assert.ok(typeof grammarCompileTime.max === "number");

    // assert that calculating statistics manually from the raw counters
    // we get the same restults as from PerfMetrics
    assert.strictEqual(
      (perfMetrics.rawMetrics.generateDurations / 1000).toFixed(3),
      generateDuration.mean.toFixed(3),
    );

    assert.strictEqual(
      (perfMetrics.rawMetrics.tokenizationDurations / 1000).toFixed(3),
      tokenizationDuration.mean.toFixed(3),
    );

    assert.strictEqual(
      (perfMetrics.rawMetrics.detokenizationDurations / 1000).toFixed(3),
      detokenizationDuration.mean.toFixed(3),
    );

    assert.ok(perfMetrics.rawMetrics.timesToFirstToken.length > 0);
    assert.ok(perfMetrics.rawMetrics.newTokenTimes.length > 0);
    assert.ok(perfMetrics.rawMetrics.tokenInferDurations.length > 0);
    assert.ok(perfMetrics.rawMetrics.batchSizes.length > 0);
    assert.ok(perfMetrics.rawMetrics.durations.length > 0);
    assert.ok(perfMetrics.rawMetrics.inferenceDurations.length > 0);
    assert.ok(perfMetrics.rawMetrics.grammarCompileTimes.length === 0);
  });

  it("test perfMetrics.add()", async () => {
    const config = {
      max_new_tokens: 5,
      return_decoded_results: true,
    };
    const res1 = await pipeline.generate("prompt1", config);
    const res2 = await pipeline.generate("prompt2", config);

    const perfMetrics1 = res1.perfMetrics;
    const perfMetrics2 = res2.perfMetrics;

    const totalNumGeneratedTokens =
      perfMetrics1.getNumGeneratedTokens() + perfMetrics2.getNumGeneratedTokens();

    perfMetrics1.add(perfMetrics2);
    assert.strictEqual(perfMetrics1.getNumGeneratedTokens(), totalNumGeneratedTokens);

    assert.throws(() => perfMetrics1.add({}), {
      message: /Passed argument is not of type PerfMetrics/,
    });
  });
});

describe("stream()", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");
  });

  it("stream() with max_new_tokens", async () => {
    const streamer = pipeline.stream("Print hello world", {
      max_new_tokens: 5,
    });
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
    }
    assert.ok(chunks.length < 5);
  });

  it("stream() with stop_strings", async () => {
    const streamer = pipeline.stream("Print hello world", {
      stop_strings: new Set(["world"]),
      include_stop_str_in_output: true,
    });
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
    }
    assert.ok(chunks[chunks.length - 1].includes("world"));
  });

  it("early break of stream", async () => {
    const streamer = pipeline.stream("Print hello world");
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
      if (chunks.length >= 5) {
        break;
      }
    }
    assert.equal(chunks.length, 5);
  });
});
