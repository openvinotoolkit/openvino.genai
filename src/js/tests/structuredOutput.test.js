import { LLMPipeline, StructuredOutputConfig } from "../dist/index.js";
import { models } from "./models.js";
import assert from "node:assert/strict";
import { describe, it, before } from "node:test";
import os from "node:os";

const INSTRUCT_MODEL_PATH =
  process.env.INSTRUCT_MODEL_PATH || `./tests/models/${models.InstructLLM.split("/")[1]}`;

describe(
  "LLMPipeline.generate() with generation config",
  // Ticket - 179439
  { skip: os.platform() === "darwin" },
  () => {
    let pipeline = null;

    before(async function () {
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

    it("generate with TriggeredTags in structural_tags_config", async () => {
      const prompt = "TriggeredTags. Repeat word 'function'";
      const generationConfig = {
        max_new_tokens: 100,
        structured_output_config: new StructuredOutputConfig({
          structural_tags_config: StructuredOutputConfig.TriggeredTags({
            triggers: ["function"],
            tags: [
              StructuredOutputConfig.Tag({
                begin: "function",
                content: StructuredOutputConfig.ConstString("A"),
                end: "</function>",
              }),
              StructuredOutputConfig.Tag({
                begin: "function",
                content: StructuredOutputConfig.ConstString("B"),
                end: "</function>",
              }),
            ],
            atLeastOne: true,
            stopAfterFirst: true,
          }),
        }),
      };

      const result = await pipeline.generate(prompt, generationConfig);
      assert.ok(
        /(function(A|B)<\/function>)/gs.test(result.toString()),
        `No matches found in output: ${result.toString()}`,
      );
    });

    it("generate with ConstString in structural_tags_config", async () => {
      const generationConfig = {
        max_new_tokens: 50,
        structured_output_config: {
          structural_tags_config: StructuredOutputConfig.ConstString("constant_string"),
        },
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
      };
      const prompt = "";
      const res = await pipeline.generate(prompt, generationConfig);
      const text = res.texts[0];
      assert.equal(text, "abc", `Unexpected output: ${text}`);
    });
  },
);
