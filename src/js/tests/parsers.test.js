// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import { LLMPipeline } from "../dist/index.js";
import { models } from "./models.js";

const MODEL_PATH = process.env.MODEL_PATH || `./tests/models/${models.LLM.split("/")[1]}`;

class PostfixParser {
  constructor(postfix) {
    this.postfix = postfix;
  }

  parse(message) {
    message.content += this.postfix;
  }
}

describe("Use parsers from js", () => {
  it("should append postfix to message content", () => {
    const postfix = "<test_postfix>";
    const parser = new PostfixParser(postfix);
    const message = { content: "Original content" };

    parser.parse(message);

    assert.strictEqual(
      message.content,
      "Original content" + postfix,
      "Postfix should be appended to message content",
    );
  });
});

describe("LLMPipeline with Parser in GenerationConfig", () => {
  let pipeline = null;

  const upperCaseParser = {
    parse: (message) => {
      message.content = message.content.toUpperCase();
    },
  };

  const postfix = "<parsed>";
  const postfixParser = new PostfixParser(postfix);

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");
  });

  it("should apply custom parser object", async () => {
    const config = {
      max_new_tokens: 10,
      parsers: [upperCaseParser],
    };

    const result = await pipeline.generate("Hello", config);
    assert.strictEqual(
      result.parsed[0].content,
      result.texts[0].toUpperCase(),
      "Parsed content should be uppercase",
    );
  });

  it("should apply custom parser class", async () => {
    const config = {
      max_new_tokens: 10,
      parsers: [postfixParser],
    };

    const result = await pipeline.generate("Hello", config);
    assert.strictEqual(
      result.parsed[0].content,
      result.texts[0] + postfix,
      "Parsed content should be updated with postfix",
    );
  });

  it("should apply multiple parsers in sequence", async () => {
    const config = {
      max_new_tokens: 10,
      parsers: [postfixParser, upperCaseParser, postfixParser],
    };

    const result = await pipeline.generate("Hello", config);
    assert.strictEqual(
      result.parsed[0].content,
      (result.texts[0] + postfix).toUpperCase() + postfix,
      "Parsers should be applied in sequence",
    );
  });

  it("should work with empty parsers array", async () => {
    const config = {
      max_new_tokens: 10,
      parsers: [],
    };

    const result = await pipeline.generate("Hello", config);
    assert.ok(result.texts.length > 0);
  });

  it("should throw error if parser is not an object", async () => {
    const config = {
      parsers: ["not_a_parser"],
    };

    await assert.rejects(
      async () => await pipeline.generate("Hello", config),
      /Parser must be a JS object with a 'parse' method/,
    );
  });

  it("should throw error if parser doesn't have parse method", async () => {
    const config = {
      parsers: [{ notParse: () => {} }],
    };

    await assert.rejects(
      async () => await pipeline.generate("Hello", config),
      /Parser object must have a 'parse' method/,
    );
  });

  it("should throw error if parse is not a function", async () => {
    const config = {
      parsers: [{ parse: "not_a_function" }],
    };

    await assert.rejects(
      async () => await pipeline.generate("Hello", config),
      /'parse' property of Parser object must be a function/,
    );
  });
});
