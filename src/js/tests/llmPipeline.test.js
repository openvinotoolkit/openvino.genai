import { ChatHistory, LLMPipeline, StructuredOutputConfig } from "../dist/index.js";
import { LLMPipeline as LLM } from "../dist/pipelines/llmPipeline.js";

import assert from "node:assert/strict";
import { describe, it, before } from "node:test";
import { hrtime } from "node:process";
import os from "node:os";

const { LLM_PATH } = process.env;

if (!LLM_PATH) {
  throw new Error("Please set LLM_PATH environment variable to run the tests.");
}

describe("LLMPipeline initialization", () => {
  it("test LLMPipeline(modelPath)", async () => {
    const pipe = await LLMPipeline(LLM_PATH);
    assert.strictEqual(typeof pipe, "object");
  });

  it("test error LLMPipeline(modelPath, LLMPipelineProperties)", async () => {
    // Test for JS. In TS it won't be possible to call LLMPipeline with wrong types.
    await assert.rejects(LLMPipeline(LLM_PATH, {}), /The second argument must be a device string./);
  });

  it("test SchedulerConfig", async (t) => {
    if (os.platform() === "darwin") {
      t.skip("only support x64 platform or ARM with SVE support");
      return;
    }
    const schedulerConfig = {
      max_num_batched_tokens: 32,
    };
    await assert.doesNotReject(LLMPipeline(LLM_PATH, "CPU", { schedulerConfig: schedulerConfig }));
  });

  it("should throw an error if pipeline is already initialized", async () => {
    const pipeline = await LLMPipeline(LLM_PATH, "CPU");

    await assert.rejects(pipeline.init(), /LLMPipeline is already initialized/);
  });

  it("should throw an error if pipeline is not initialized", async () => {
    const pipeline = new LLM(LLM_PATH, "CPU");

    await assert.rejects(pipeline.generate("prompt"), /LLMPipeline is not initialized/);
  });

  it("getGenerationConfig throws if pipeline not initialized", () => {
    const uninitPipeline = new LLM(LLM_PATH, "CPU");
    assert.throws(() => uninitPipeline.getGenerationConfig(), /LLMPipeline is not initialized/);
  });

  it("setGenerationConfig throws if pipeline not initialized", () => {
    const uninitPipeline = new LLM(LLM_PATH, "CPU");
    assert.throws(
      () => uninitPipeline.setGenerationConfig({ max_new_tokens: 10 }),
      /LLMPipeline is not initialized/,
    );
  });
});

describe("General LLM test", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(LLM_PATH, "CPU");
  });

  describe("LLMPipeline methods", () => {
    it("should generate non empty result", async () => {
      const result = await pipeline.generate(
        "Type something in English",
        { temperature: 0, max_new_tokens: 4 },
        () => {},
      );

      assert.ok(result.texts.length > 0);
      assert.strictEqual(typeof result.texts[0], "string");
    });

    it("should include tokenizer", async () => {
      const tokenizer = pipeline.getTokenizer();
      assert.strictEqual(typeof tokenizer, "object");
    });

    it("getGenerationConfig returns object with expected fields", async () => {
      const config = pipeline.getGenerationConfig();
      const result = await pipeline.generate("What is OpenVINO?", { ...config, max_new_tokens: 5 });
      assert.strictEqual(result.texts.length, 1);
    });

    it("setGenerationConfig updates config, getGenerationConfig returns updated values", async () => {
      const original = pipeline.getGenerationConfig();
      const originalMaxNewTokens = original.max_new_tokens;
      const config = { ...original, max_new_tokens: 5 };
      pipeline.setGenerationConfig(config);
      const newConfig = pipeline.getGenerationConfig();
      assert.deepEqual(config, newConfig);
      assert.notEqual(newConfig.max_new_tokens, originalMaxNewTokens);
      const result = await pipeline.generate("What is OpenVINO?", newConfig);
      assert.strictEqual(result.texts.length, 1);
    });

    it("should throw an error if pipeline is already initialized", async () => {
      await assert.rejects(async () => await pipeline.init(), /LLMPipeline is already initialized/);
    });
  });

  describe("generation parameters validation", () => {
    it("should throw an error if no arguments passed to generate", async () => {
      await assert.rejects(
        async () => await pipeline.generate(),
        /Passed argument must be a string, ChatHistory or an array of strings./,
      );
    });

    it("should throw an error if generationCallback is not a function", async () => {
      await assert.rejects(async () => await pipeline.generate("prompt", {}, false), {
        name: "Error",
        message: "Streamer must be a function",
      });
    });

    it("should throw an error if options specified but not an object", async () => {
      await assert.rejects(async () => await pipeline.generate("prompt", "options", () => {}), {
        name: "Error",
        message: "Options must be an object",
      });
    });

    it("should perform generation with default options", async () => {
      await assert.doesNotReject(pipeline.generate("prompt", { max_new_tokens: 1 }));
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
      assert.strictEqual(result.texts.length, 1);
      assert.ok(result.texts[0].length > 0);
    });
  });

  describe("stream()", () => {
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
      const generationConfig = {
        stop_strings: new Set(["C"]),
        include_stop_str_in_output: true,
        max_new_tokens: 100,
        structured_output_config: {
          structural_tags_config: StructuredOutputConfig.Concat(
            StructuredOutputConfig.ConstString("_A_"),
            StructuredOutputConfig.ConstString("_B_"),
            StructuredOutputConfig.ConstString("_C_"),
            StructuredOutputConfig.ConstString("_D_"),
            StructuredOutputConfig.ConstString("_E_"),
          ),
        },
      };
      const prompt = "Print alphabet";
      const streamer = pipeline.stream(prompt, generationConfig);
      const chunks = [];
      for await (const chunk of streamer) {
        chunks.push(chunk);
      }
      assert.ok(chunks[chunks.length - 1].includes("C"));

      generationConfig.include_stop_str_in_output = false;
      const streamer2 = pipeline.stream(prompt, generationConfig);
      const chunks2 = [];
      for await (const chunk of streamer2) {
        chunks2.push(chunk);
      }
      assert.ok(!chunks2[chunks2.length - 1].includes("C"));
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

    it("stream() with array of strings", async () => {
      assert.throws(() => {
        pipeline.stream(["prompt1", "prompt2", "prompt3"]);
      }, /Streaming is not supported for array of inputs/);
    });
  });
});

describe("LLMPipeline in chat mode", () => {
  let pipeline = null;
  // We need to keep previous messages between tests to avoid error for SDPA backend (macOS in CI)
  const chatHistory = new ChatHistory();

  before(async () => {
    pipeline = await LLMPipeline(LLM_PATH, "CPU", { ATTENTION_BACKEND: "SDPA" });
  });

  it("generate(chatHistory, config)", async () => {
    chatHistory.setMessages([
      { role: "user", content: "Hello!" },
      { role: "assistant", content: "Hi! How can I help you?" },
      { role: "user", content: "Tell me a joke." },
    ]);
    const config = {
      max_new_tokens: 10,
    };
    const reply = await pipeline.generate(chatHistory, config);
    // We need to keep previous messages between tests to avoid error for SDPA backend (macOS in CI)
    chatHistory.push({ role: "assistant", content: reply.toString() });
    assert.ok(Array.isArray(reply.texts));
    assert.equal(reply.texts.length, 1);
    assert.ok(typeof reply.texts[0] === "string");
    console.log("Reply:", reply.toString());
  });

  it("generate(chatHistory, config) with invalid chat history", async () => {
    const chatHistory = [1, "assistant", null];
    const config = {
      max_new_tokens: 10,
    };
    await assert.rejects(async () => {
      await pipeline.generate(chatHistory, config);
    }, /An incorrect input value has been passed./);
  });

  it("stream(chatHistory, config)", async () => {
    chatHistory.push({ role: "user", content: "Tell me another joke." });
    const config = {
      max_new_tokens: 10,
    };
    const streamer = await pipeline.stream(chatHistory, config);
    const chunks = [];
    for await (const chunk of streamer) {
      chunks.push(chunk);
    }
    assert.ok(chunks.length > 0);
    // We need to keep previous messages between tests to avoid error for SDPA backend (macOS in CI)
    chatHistory.push({ role: "assistant", content: chunks.join("") });
  });

  it("startChat(systemMessage) then generate() then finishChat()", async () => {
    await pipeline.startChat("You are a helpful assistant.");
    const config = { max_new_tokens: 10 };
    const result = await pipeline.generate("Reply with one word: hello", config);
    const result2 = await pipeline.generate("Reply with one word: bye", config);
    await pipeline.finishChat();
    assert.ok(result.texts[0].length > 0);
    assert.ok(result2.texts[0].length > 0);
  });
});

describe("LLMPipeline perf metrics", () => {
  it("DecodedResults.perfMetrics", async (t) => {
    if (os.platform() === "darwin") {
      t.skip("Skipping perfMetrics test on macOS. Ticket - 173286");
      return;
    }

    const config = {
      max_new_tokens: 20,
    };
    const prompt = "The Sky is blue because";
    const start = hrtime.bigint();
    const pipeline = await LLMPipeline(LLM_PATH, "CPU");
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
    //
    // Disabled due to potential floating-point differences. (CVS-175568)
    // assert.strictEqual(
    //   (perfMetrics.rawMetrics.generateDurations / 1000).toFixed(3),
    //   generateDuration.mean.toFixed(3),
    // );

    // assert.strictEqual(
    //   (perfMetrics.rawMetrics.tokenizationDurations / 1000).toFixed(3),
    //   tokenizationDuration.mean.toFixed(3),
    // );

    // assert.strictEqual(
    //   (perfMetrics.rawMetrics.detokenizationDurations / 1000).toFixed(3),
    //   detokenizationDuration.mean.toFixed(3),
    // );

    assert.ok(perfMetrics.rawMetrics.timesToFirstToken.length > 0);
    assert.ok(perfMetrics.rawMetrics.newTokenTimes.length > 0);
    assert.ok(perfMetrics.rawMetrics.tokenInferDurations.length > 0);
    assert.ok(perfMetrics.rawMetrics.batchSizes.length > 0);
    assert.ok(perfMetrics.rawMetrics.durations.length > 0);
    assert.ok(perfMetrics.rawMetrics.inferenceDurations.length > 0);
    assert.ok(perfMetrics.rawMetrics.grammarCompileTimes.length === 0);
  });

  it("test perfMetrics.add()", async () => {
    const pipeline = await LLMPipeline(LLM_PATH, "CPU");
    const config = {
      max_new_tokens: 5,
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
