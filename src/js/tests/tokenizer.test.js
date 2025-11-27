import { LLMPipeline, ChatHistory, Tokenizer } from "../dist/index.js";

import assert from "node:assert/strict";
import { describe, it, before, after } from "node:test";
import { models } from "./models.js";
import fs from "node:fs/promises";
import { join } from "node:path";
import { addon as ovAddon } from "openvino-node";

const MODEL_PATH = process.env.MODEL_PATH || `./tests/models/${models.LLM.split("/")[1]}`;

describe("tokenizer constructors", () => {
  it("tokenizer constructors with one argument", () => {
    const tokenizer = new Tokenizer(MODEL_PATH);

    assert.ok(tokenizer);
  });

  it("tokenizer constructors with multiple arguments", async () => {
    const tokenizerName = join(MODEL_PATH, "openvino_tokenizer");
    const detokenizerName = join(MODEL_PATH, "openvino_detokenizer");
    const tokenizerModel = await fs.readFile(`${tokenizerName}.xml`, "utf8");
    const tokenizerWeights = await fs.readFile(`${tokenizerName}.bin`);
    const detokenizerModel = await fs.readFile(`${detokenizerName}.xml`, "utf8");
    const detokenizerWeights = await fs.readFile(`${detokenizerName}.bin`);

    const tokenizerTensor = new ovAddon.Tensor("u8", [tokenizerWeights.length], tokenizerWeights);
    const detokenizerTensor = new ovAddon.Tensor(
      "u8",
      [detokenizerWeights.length],
      detokenizerWeights,
    );

    const tokenizer = new Tokenizer(
      tokenizerModel,
      tokenizerTensor,
      detokenizerModel,
      detokenizerTensor,
    );

    assert.ok(tokenizer);
  });
});

describe("tokenizer functions", async () => {
  let pipeline = null;
  let tokenizer = null;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");

    await pipeline.startChat();
    tokenizer = pipeline.getTokenizer();
  });

  after(async () => {
    await pipeline.finishChat();
  });

  it("applyChatTemplate return string", () => {
    const template = tokenizer.applyChatTemplate(
      [
        {
          role: "user",
          content: "continue: 1 2 3",
        },
      ],
      false,
    );
    assert.strictEqual(typeof template, "string");
  });

  it("applyChatTemplate with chat history", () => {
    const chatHistory = new ChatHistory([
      {
        role: "user",
        content: "continue: 1 2 3",
      },
    ]);
    const template = tokenizer.applyChatTemplate(chatHistory, false);
    assert.strictEqual(typeof template, "string");
  });

  it("applyChatTemplate with true addGenerationPrompt", () => {
    const template = tokenizer.applyChatTemplate(
      [
        {
          role: "user",
          content: "continue: 1 2 3",
        },
      ],
      true,
    );
    assert.ok(template.includes("assistant"));
  });

  it("applyChatTemplate with missed addGenerationPrompt", () => {
    assert.throws(() =>
      tokenizer.applyChatTemplate([
        {
          role: "user",
          content: "continue: 1 2 3",
        },
      ]),
    );
  });

  it("applyChatTemplate with incorrect type of history", () => {
    assert.throws(() => tokenizer.applyChatTemplate("prompt", false));
  });

  it("applyChatTemplate with unknown property", () => {
    const testValue = "1234567890";
    const template = tokenizer.applyChatTemplate(
      [
        {
          role: "user",
          content: "continue: 1 2 3",
          unknownProp: testValue,
        },
      ],
      false,
    );
    assert.ok(!template.includes(testValue));
  });

  it("applyChatTemplate use custom chatTemplate", () => {
    const prompt = "continue: 1 2 3";
    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% endfor %}`;
    const template = tokenizer.applyChatTemplate(
      [
        {
          role: "user",
          content: prompt,
        },
      ],
      false,
      chatTemplate,
    );
    assert.strictEqual(template, `${prompt}\n`);
  });

  it("applyChatTemplate use tools", () => {
    const prompt = "question";
    const chatHistory = [
      {
        role: "user",
        content: prompt,
      },
    ];
    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% for tool in tools %}{{ tool | tojson }}{% endfor %}
{% endfor %}`;
    const tools = [{ type: "function", function: { name: "test" } }];
    const templatedHistory = tokenizer.applyChatTemplate(chatHistory, false, chatTemplate, tools);
    const expected = `${prompt}\n{"type": "function", "function": {"name": "test"}}`;
    assert.strictEqual(templatedHistory, expected);
  });

  it("applyChatTemplate use tool from chat history", () => {
    const prompt = "question";
    const chatHistory = new ChatHistory();
    chatHistory.push({ role: "user", content: prompt });
    chatHistory.setTools([{ type: "function", function: { name: "test" } }]);

    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% for tool in tools %}{{ tool | tojson }}{% endfor %}
{% endfor %}`;
    const templatedHistory = tokenizer.applyChatTemplate(chatHistory, false, chatTemplate);
    const expected = `${prompt}\n{"type": "function", "function": {"name": "test"}}`;
    assert.strictEqual(templatedHistory, expected);
  });

  it("applyChatTemplate use extra_context", () => {
    const prompt = "question";
    const chatHistory = [
      {
        role: "user",
        content: prompt,
      },
    ];
    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% if enable_thinking is defined and enable_thinking is false %}No thinking{% endif %}
{% endfor %}`;
    const tools = [];
    // eslint-disable-next-line camelcase
    const extraContext = { enable_thinking: false };
    const templatedHistory = tokenizer.applyChatTemplate(
      chatHistory,
      false,
      chatTemplate,
      tools,
      extraContext,
    );
    const expected = `${prompt}\nNo thinking`;
    assert.strictEqual(templatedHistory, expected);
  });

  it("applyChatTemplate use extra_context from chat history", () => {
    const prompt = "question";
    const chatHistory = new ChatHistory();
    chatHistory.push({ role: "user", content: prompt });
    // eslint-disable-next-line camelcase
    chatHistory.setExtraContext({ enable_thinking: false });

    const chatTemplate = `{% for message in messages %}
{{ message['content'] }}
{% if enable_thinking is defined and enable_thinking is false %}No thinking{% endif %}
{% endfor %}`;
    const templatedHistory = tokenizer.applyChatTemplate(chatHistory, false, chatTemplate);
    const expected = `${prompt}\nNo thinking`;
    assert.strictEqual(templatedHistory, expected);
  });

  it("getBosToken return string", () => {
    const token = tokenizer.getBosToken();
    assert.strictEqual(typeof token, "string");
  });

  it("getBosTokenId return number", () => {
    const token = tokenizer.getBosTokenId();
    assert.strictEqual(typeof token, "bigint");
  });

  it("getEosToken return string", () => {
    const token = tokenizer.getEosToken();
    assert.strictEqual(typeof token, "string");
  });

  it("getEosTokenId return number", () => {
    const token = tokenizer.getEosTokenId();
    assert.strictEqual(typeof token, "bigint");
  });

  it("getPadToken return string", () => {
    const token = tokenizer.getPadToken();
    assert.strictEqual(typeof token, "string");
  });

  it("getPadTokenId return number", () => {
    const token = tokenizer.getPadTokenId();
    assert.strictEqual(typeof token, "bigint");
  });

  it("setChatTemplate updates template", () => {
    const originalTemplate = tokenizer.getChatTemplate();
    assert.strictEqual(typeof originalTemplate, "string");

    const customTemplate = "Custom template: {{ messages }}";
    tokenizer.setChatTemplate(customTemplate);

    const updatedTemplate = tokenizer.getChatTemplate();
    assert.strictEqual(updatedTemplate, customTemplate);

    // Restore original template
    tokenizer.setChatTemplate(originalTemplate);
  });

  // Fix getOriginalChatTemplate issue CVS-176638
  it.skip("getOriginalChatTemplate returns the original string", () => {
    const originalTemplate = tokenizer.getChatTemplate();
    tokenizer.setChatTemplate("Custom template: {{ messages }}");

    const template = tokenizer.getOriginalChatTemplate();
    assert.strictEqual(template, originalTemplate);

    // Restore original template
    tokenizer.setChatTemplate(originalTemplate);
  });

  it("encode single string returns TokenizedInputs", () => {
    const text = "Hello world";
    const result = tokenizer.encode(text);

    assert.ok(result.input_ids, "Should have input_ids");
    assert.ok(result.attention_mask, "Should have attention_mask");
    assert.strictEqual(typeof result.input_ids, "object");
    assert.strictEqual(typeof result.attention_mask, "object");
  });

  it("encode with options", () => {
    const text = "Hello world";
    const result = tokenizer.encode(text, {
      add_special_tokens: false,
      pad_to_max_length: true,
      max_length: 1000,
      padding_side: "left",
    });
    // const padTokenId = tokenizer.getPadTokenId();

    assert.ok(result.input_ids);
    assert.strictEqual(
      result.input_ids.getShape()[1],
      1000,
      "input_ids should be padded to maxLength",
    );
    // Uncomment after fixing padding issue CVS-176636
    // assert.strictEqual(
    //   result.input_ids.getData()[0],
    //   padTokenId,
    //   "input_ids should be left padded",
    // );
  });

  it("encode array of strings", () => {
    const texts = ["Hello", "World"];
    const result = tokenizer.encode(texts);

    assert.strictEqual(result.input_ids.getShape()[0], texts.length);
    assert.strictEqual(result.attention_mask.getShape()[0], 2);
  });

  it("decode array of token IDs to string", () => {
    const tokenIds = [1, 2, 3];
    const decoded = tokenizer.decode(tokenIds);

    assert.strictEqual(typeof decoded, "string");
  });

  // Fix skip_special_tokens functionality CVS-176639
  it.skip("decode with skip_special_tokens option", () => {
    const eos = tokenizer.getEosToken();
    const eosId = tokenizer.getEosTokenId();
    const tokenIds = [10n, 20n, 30n, eosId];
    const decoded1 = tokenizer.decode(tokenIds, { skip_special_tokens: true });
    const decoded2 = tokenizer.decode(tokenIds, { skip_special_tokens: false });

    assert.strictEqual(typeof decoded1, "string");
    assert.strictEqual(typeof decoded2, "string");
    assert.strictEqual(decoded2, decoded1 + eos);
  });

  it("decode batch of token sequences", () => {
    const batchTokens = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const decoded = tokenizer.decode(batchTokens);

    assert.strictEqual(decoded.length, 2);
  });

  it("encode and decode round trip", () => {
    const originalText = "Hello world";
    const encoded = tokenizer.encode(originalText);
    const decodedText = tokenizer.decode(encoded.input_ids);

    assert.deepEqual(decodedText, [originalText]);
  });
});

// Add model with paired input support CVS-176639
describe.skip("tokenizer with paired input", () => {
  let tokenizer = null;

  before(async () => {
    tokenizer = new Tokenizer(MODEL_PATH, { add_second_input: true, number_of_inputs: 2 });
  });

  it("supportsPairedInput return boolean", () => {
    const result = tokenizer.supportsPairedInput();

    assert.strictEqual(result, true);
  });

  it("encode paired prompts (two arrays)", () => {
    const prompts1 = ["Question 1", "Question 2"];
    const prompts2 = ["Answer 1", "Answer 2"];
    const result = tokenizer.encode(prompts1, prompts2);

    assert.strictEqual(result.input_ids.getShape()[0], prompts1.length);
    assert.strictEqual(result.attention_mask.getShape()[0], prompts1.length);
  });

  it("encode paired prompts (array of pairs)", () => {
    const pairs = [
      ["Question 1", "Answer 1"],
      ["Question 2", "Answer 2"],
    ];
    const result = tokenizer.encode(pairs);

    assert.strictEqual(result.input_ids.getShape()[0], pairs.length);
    assert.strictEqual(result.attention_mask.getShape()[0], pairs.length);
  });

  it("encode paired prompts broadcasting second array", () => {
    const prompts1 = ["Question 1", "Question 2", "Question 3"]; // batch size 3
    const prompts2 = ["Single answer"]; // will be broadcast
    const result = tokenizer.encode(prompts1, prompts2);

    assert.strictEqual(result.input_ids.getShape()[0], prompts1.length);
    assert.strictEqual(result.attention_mask.getShape()[0], prompts1.length);
  });
});
