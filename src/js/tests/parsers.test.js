// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import {
  DeepSeekR1ReasoningParser,
  Llama3PythonicToolParser,
  Llama3JsonToolParser,
  LLMPipeline,
  Phi4ReasoningParser,
  ReasoningParser,
} from "../dist/index.js";
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

class CustomReasoningParser extends ReasoningParser {
  parse(message) {
    super.parse(message);
    // Append custom text to reasoning content
    message.reasoning_content += "[custom processed]";
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

  describe("ReasoningParser", () => {
    it("should parse message and extract reasoning content", () => {
      const parser = new ReasoningParser();
      const message = {
        content: "Before <think>reasoning content</think> After",
      };

      parser.parse(message);

      assert.strictEqual(message.reasoning_content, "reasoning content");
    });

    it("should parse without opening tag", () => {
      const parser = new ReasoningParser({ expectOpenTag: false });
      const message = {
        content: "Thinking</think>Answer",
      };

      parser.parse(message);
      // CVS-178738 The ReasoningParser incorrectly works with expect_open_tag=False
      // Should be strictEqual but currently not working
      assert.notStrictEqual(message.reasoning_content, "Thinking");
    });

    it("should keep original content when keepOriginalContent is true", () => {
      const parser = new ReasoningParser({
        keepOriginalContent: false,
      });
      const message = {
        content: "Before <think>reasoning content</think> After",
      };

      parser.parse(message);

      assert.strictEqual(message.reasoning_content, "reasoning content");
      assert.strictEqual(message.content, "Before  After");
    });

    it("should work with custom tags", () => {
      const parser = new ReasoningParser({
        openTag: "<custom>",
        closeTag: "</custom>",
      });
      const message = {
        content: "Text <custom>custom reasoning</custom> more text",
      };

      parser.parse(message);

      assert.strictEqual(message.reasoning_content, "custom reasoning");
    });

    it("subclass should extend functionality", () => {
      const parser = new CustomReasoningParser();
      const message = {
        content: "Info <think>some reasoning</think> end",
      };

      parser.parse(message);

      assert.strictEqual(message.reasoning_content, "some reasoning[custom processed]");
    });
  });

  it("DeepSeekR1ReasoningParser should work", () => {
    const deepSeekParser = new DeepSeekR1ReasoningParser();
    const reasoning = "DeepSeek reasoning";
    const message = {
      content: `${reasoning}</think> After`,
    };
    deepSeekParser.parse(message);
    // CVS-178737 DeepSeekR1ReasoningParser does not extract reasoning content correctly
    // Should be strictEqual but currently not working
    assert.notStrictEqual(message.reasoning_content, reasoning);
  });

  it("Phi4ReasoningParser should work", () => {
    const phi4Parser = new Phi4ReasoningParser();
    const reasoning = "Phi4 reasoning";
    const message = {
      content: `Before <think>${reasoning}</think> After`,
    };
    phi4Parser.parse(message);
    assert.strictEqual(message.reasoning_content, reasoning);
  });

  it('Llama3PythonicToolParser should extract tool usage between "<tool>" tags', () => {
    const llama3Parser = new Llama3PythonicToolParser();
    const toolCalling = '[get_weather(location="New York, NY", unit="celsius")]';
    const message = {
      content: `Call this tool ${toolCalling} to get answers.`,
    };
    llama3Parser.parse(message);
    assert.deepStrictEqual(message.tool_calls, [
      {
        arguments: {
          location: "New York, NY",
          unit: "celsius",
        },
        name: "get_weather",
      },
    ]);
  });

  it("Llama3JsonToolParser should extract JSON tool calls", () => {
    const llama3JsonToolParser = new Llama3JsonToolParser();
    const toolCalling = {
      function: {
        name: "get_weather",
        parameters: {
          location: "New York, NY",
          unit: "celsius",
        },
      },
      type: "function",
    };
    const message = {
      content: `I'll help you with that. ${JSON.stringify(toolCalling)}`,
    };
    llama3JsonToolParser.parse(message);
    assert.deepStrictEqual(message.tool_calls, [toolCalling]);
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

describe("LLMPipeline with ReasoningParser", () => {
  let pipeline, postfixParser;

  before(async () => {
    pipeline = await LLMPipeline(MODEL_PATH, "CPU");
    postfixParser = new PostfixParser("<think>reasoning</think>");
  });

  it("should accept ReasoningParser in parsers array", async () => {
    const parser = new ReasoningParser({
      openTag: "```python\n",
      closeTag: "```\n",
    });
    const config = {
      max_new_tokens: 100,
      parsers: [parser],
    };

    const result = await pipeline.generate("Create python code to print 'Hello, World'", config);
    assert.ok(
      result.parsed[0].reasoning_content?.length > 0,
      "Reasoning content should be extracted",
    );
  });

  it("should work with multiple parsers including ReasoningParser", async () => {
    const reasoningParser = new ReasoningParser();
    const postfixParser = new PostfixParser("<think>reasoning</think>");

    const config = {
      max_new_tokens: 10,
      parsers: [postfixParser, reasoningParser],
    };

    const result = await pipeline.generate("Create python code to print 'Hello, World'", config);
    assert.strictEqual(
      result.parsed[0].reasoning_content,
      "reasoning",
      "Reasoning content should be extracted after postfix parser",
    );
  });

  it("should accept subclass of ReasoningParser", async () => {
    const parser = new CustomReasoningParser();
    const config = {
      max_new_tokens: 100,
      parsers: [postfixParser, parser],
    };

    const result = await pipeline.generate("Explain the theory of relativity", config);
    assert.strictEqual(
      result.parsed[0].reasoning_content,
      "reasoning[custom processed]",
      "Custom processing should be applied to reasoning content",
    );
  });
});
