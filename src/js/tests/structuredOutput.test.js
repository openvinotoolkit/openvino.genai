import { LLMPipeline, StructuredOutputConfig } from "../dist/index.js";
import { models } from "./models.js";
import assert from "node:assert/strict";
import { describe, it, before } from "node:test";
import os from "node:os";

const INSTRUCT_MODEL_PATH =
  process.env.INSTRUCT_MODEL_PATH || `./tests/models/${models.InstructLLM.split("/")[1]}`;

describe("LLMPipeline.generate() with generation config", () => {
  let pipeline = null;

  before(async () => {
    pipeline = await LLMPipeline(INSTRUCT_MODEL_PATH, "CPU");
  });

  it("generate with json schema in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        json_schema: JSON.stringify({
          type: "object",
          properties: {
            name: { type: "string" },
            age: { type: "number" },
            city: { type: "string" },
          },
          required: ["name", "age", "city"],
        }),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      assert.fail(`Failed to parse JSON: ${text}`);
    }
    assert.ok(typeof parsed === "object");
    assert.ok(typeof parsed.name === "string");
    assert.ok(typeof parsed.age === "number");
    assert.ok(typeof parsed.city === "string");
  });

  it("generate with StructuredOutputConfig.JSONSchema in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.JSONSchema(
          JSON.stringify({
            type: "object",
            properties: {
              name: { type: "string" },
              age: { type: "number" },
              city: { type: "string" },
            },
            required: ["name", "age", "city"],
          }),
        ),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      assert.fail(`Failed to parse JSON: ${text}`);
    }
    assert.ok(typeof parsed === "object");
    assert.ok(typeof parsed.name === "string");
    assert.ok(typeof parsed.age === "number");
    assert.ok(typeof parsed.city === "string");
  });

  it("generate with regex in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 10,
      structured_output_config: {
        regex: "yes|no",
      },
      return_decoded_results: true,
    };
    const prompt = `Answer the question with "yes" or "no": Is the sky blue?`;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim().toLowerCase();
    assert.ok(text === "yes" || text === "no", `Unexpected answer: ${text}`);
  });

  it("generate with StructuredOutputConfig.Regex in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 10,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.Regex("yes|no"),
      },
      return_decoded_results: true,
    };
    const prompt = `Answer the question with "yes" or "no": Is the sky blue?`;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim().toLowerCase();
    assert.ok(text === "yes" || text === "no", `Unexpected answer: ${text}`);
  });

  it("generate with StructuredOutputConfig.Union in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 10,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.Union(
          StructuredOutputConfig.Regex("yes"),
          StructuredOutputConfig.Regex("no"),
        ),
      },
      return_decoded_results: true,
    };
    const prompt = `Answer the question with "yes" or "no": Is the sky blue?`;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim().toLowerCase();
    assert.ok(text === "yes" || text === "no", `Unexpected answer: ${text}`);
  });

  it("generate with grammar in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        grammar: `root::= "SELECT " column (", " column)? " from " table ";"
column::= "name" | "username" | "email" | "postcode" | "*"
table::= "users" | "orders" | "products"`,
      },
      return_decoded_results: true,
    };
    const prompt = `"Respond with a SQL query using the grammar. Generate an SQL query to show the 'username' and 'email' from the 'users' table."`;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim();
    assert.equal(text, "SELECT username, email from users;", `Unexpected format: ${text}`);
  });

  it("generate with StructuredOutputConfig.EBNF in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config:
          StructuredOutputConfig.EBNF(`root::= "SELECT " column (", " column)? " from " table ";"
column::= "name" | "username" | "email" | "postcode" | "*"
table::= "users" | "orders" | "products"`),
      },
      return_decoded_results: true,
    };
    const prompt = `"Respond with a SQL query using the grammar. Generate an SQL query to show the 'username' and 'email' from the 'users' table."`;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim();
    assert.equal(text, "SELECT username, email from users;", `Unexpected format: ${text}`);
  });

  it("generate with StructuredOutputConfig.Concat in structured_output_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.Concat(
          StructuredOutputConfig.JSONSchema(
            JSON.stringify({
              type: "object",
              properties: {
                name: { type: "string" },
                age: { type: "number" },
                city: { type: "string" },
              },
              required: ["name", "age", "city"],
            }),
          ),
          StructuredOutputConfig.Union(
            StructuredOutputConfig.Regex("A"),
            StructuredOutputConfig.Regex("B"),
          ),
        ),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0].trim();

    const postfix = text[text.length - 1];
    assert.ok(postfix === "A" || postfix === "B", `Unexpected postfix: ${postfix}`);

    const jsonPart = text.substring(0, text.length - 1);
    let parsed;
    try {
      parsed = JSON.parse(jsonPart);
    } catch {
      assert.fail(`Failed to parse JSON: ${text}`);
    }
    assert.ok(typeof parsed === "object");
    assert.ok(typeof parsed.name === "string");
    assert.ok(typeof parsed.age === "number");
    assert.ok(typeof parsed.city === "string");
  });

  it("generate with TriggeredTags in structural_tags_config", async (testContext) => {
    if (os.platform() === "darwin" || os.platform() === "win32") {
      testContext.skip("Skipped due to inconsistent LLM outputs. CVS-175278");
      return;
    }
    const tools = [
      {
        name: "get_weather",
        schema: {
          type: "object",
          properties: {
            location: { type: "string" },
            unit: { type: "string", enum: ["metric", "imperial"] },
          },
          required: ["location", "unit"],
        },
      },
      {
        name: "get_currency_exchange",
        schema: {
          type: "object",
          properties: {
            fromCurrency: { type: "string" },
            toCurrency: { type: "string" },
            amount: { type: "number" },
          },
          required: ["fromCurrency", "toCurrency", "amount"],
        },
      },
    ];
    const generationConfig = {
      max_new_tokens: 200,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.TriggeredTags({
          triggers: ["<function="],
          tags: tools.map((tool) =>
            StructuredOutputConfig.Tag({
              begin: `<function="${tool.name}">`,
              content: StructuredOutputConfig.JSONSchema(JSON.stringify(tool.schema)),
              end: "</function>",
            }),
          ),
        }),
      },
      return_decoded_results: true,
    };
    const sysMessage =
      "You are a helpful assistant that can provide weather information and currency exchange rates. " +
      `Today is ${new Date().toISOString().split("T")[0]}. ` +
      "You can respond in natural language, always start your answer with appropriate greeting, " +
      "If you need additional information to respond you can request it by calling particular tool with structured JSON. " +
      `You can use the following tools:
${tools.map((tool) => `<function_name="${tool.name}">, arguments=${JSON.stringify(tool.schema.required)}`).join("\n")}
Please, only use the following format for tool calling in your responses:
<function="function_name">{"argument1": "value1", ...}</function>
Use the tool name and arguments as defined in the tool schema.
If you don't know the answer, just say that you don't know, but try to call the tool if it helps to answer the question.`;

    const prompt =
      "What is the weather in London today and in Paris yesterday with metric units, and how many pounds can I get for 100 euros?";
    await pipeline.startChat(sysMessage);
    try {
      const res = await pipeline.generate(prompt, generationConfig);
      const text = res.texts[0].trim();
      const matches = [...text.matchAll(/<function="([^"]+)">(.*?)<\/function>/gs)];
      assert.equal(matches.length, 3, `Expected 3 function calls, got ${matches.length}`);
      assert.ok(
        matches[0][1] === "get_weather" || matches[0][1] === "get_currency_exchange",
        `Unexpected function name: ${matches[0][1]}`,
      );
      const args = JSON.parse(matches[0][2]);
      if (matches[0][1] === "get_weather") {
        assert.ok(typeof args.location === "string", `Unexpected location: ${args.location}`);
        assert.ok(
          args.unit === "metric" || args.unit === "imperial",
          `Unexpected unit: ${args.unit}`,
        );
      } else if (matches[0][1] === "get_currency_exchange") {
        assert.ok(
          typeof args.fromCurrency === "string",
          `Unexpected fromCurrency: ${args.fromCurrency}`,
        );
        assert.ok(typeof args.toCurrency === "string", `Unexpected toCurrency: ${args.toCurrency}`);
        assert.ok(typeof args.amount === "number", `Unexpected amount: ${args.amount}`);
      }
    } finally {
      await pipeline.finishChat();
    }
  });

  it("generate with ConstString in structural_tags_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.ConstString("constant_string"),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    assert.equal(text, "constant_string", `Unexpected output: ${text}`);
  });

  it("generate with ConstString in structural_tags_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.AnyText(),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    assert.ok(text.length > 0, `Unexpected output: ${text}`);
  });

  it("generate with Tag in structural_tags_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.Tag({
          begin: "<start>",
          content: StructuredOutputConfig.ConstString("..."),
          end: "</end>",
        }),
      },
      return_decoded_results: true,
    };
    const prompt = `Generate a JSON object with the following properties:
    - name: a random name
    - age: a random age between 1 and 100
    - city: a random city
    The JSON object should be in the following format:
    {
      "name": "John Doe",
      "age": 30,
      "city": "New York"
    }
    `;
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    assert.equal(text, "<start>...</end>", `Unexpected output: ${text}`);
  });

  it("generate with QwenXMLParametersFormat in structural_tags_config", async (testContext) => {
    if (os.platform() === "darwin") {
      testContext.skip("Skipped for macOS due to inconsistent LLM outputs. CVS-175278");
      return;
    }
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.QwenXMLParametersFormat(
          JSON.stringify({
            properties: {
              status: {
                enum: ["success", "error"],
                title: "Status",
                type: "string",
              },
              data: {
                pattern: "^[A-Z][a-z]{1,20}$",
                title: "Data",
                type: "string",
              },
            },
            required: ["status", "data"],
            title: "RESTAPIResponse",
            type: "object",
          }),
        ),
      },
      return_decoded_results: true,
    };
    const prompt = "Make a request to a REST API.";
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    const statusPattern = /<parameter=status>\s*"(success|error)"\s*<\/parameter>/;
    const dataPattern = /<parameter=data>\s*[A-Z][a-z]{1,20}\s*<\/parameter>/;

    assert.ok(statusPattern.test(text), `Unexpected output: ${text}`);
    assert.ok(dataPattern.test(text), `Unexpected output: ${text}`);
  });

  it("generate with TagsWithSeparator in structural_tags_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: StructuredOutputConfig.TagsWithSeparator({
          tags: [
            StructuredOutputConfig.Tag({
              begin: "<f>",
              content: StructuredOutputConfig.ConstString("A"),
              end: "</f>",
            }),
            StructuredOutputConfig.Tag({
              begin: "<f>",
              content: StructuredOutputConfig.ConstString("B"),
              end: "</f>",
            }),
          ],
          separator: ";",
          atLeastOne: true,
          stopAfterFirst: false,
        }),
      },
      return_decoded_results: true,
    };
    const prompt = "";
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    const pattern = /(<f>(A|B)<\/f>(;<f>(A|B)<\/f>))*/;

    assert.ok(pattern.test(text), 1, `Unexpected output: ${text}`);
  });

  it("generate with string structural_tags_config", async () => {
    const generationConfig = {
      max_new_tokens: 50,
      structured_output_config: {
        structural_tags_config: JSON.stringify({
          type: "structural_tag",
          format: {
            type: "const_string",
            value: "abc",
          },
        }),
      },
      return_decoded_results: true,
    };
    const prompt = "";
    const res = await pipeline.generate(prompt, generationConfig);
    const text = res.texts[0];
    assert.equal(text, "abc", `Unexpected output: ${text}`);
  });
});
