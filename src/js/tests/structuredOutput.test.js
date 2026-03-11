import { LLMPipeline, StructuredOutputConfig } from "../dist/index.js";
import assert from "node:assert/strict";
import { describe, it, before } from "node:test";
import os from "node:os";

const { LLM_PATH } = process.env;

if (!LLM_PATH) {
  throw new Error("Please set LLM_PATH environment variable to run the tests.");
}

describe(
  "LLMPipeline.generate() with generation config",
  // Ticket - 179439
  { skip: os.platform() === "darwin" },
  () => {
    let pipeline = null;

    const person = {
      properties: {
        name: { pattern: "^[A-Z][a-z]{1,20}$", type: "string" },
        age: { maximum: 128, minimum: 0, type: "integer" },
        city: { enum: ["Dublin", "Dubai", "Munich"], type: "string" },
      },
      required: ["name", "age", "city"],
      type: "object",
    };

    before(async function () {
      pipeline = await LLMPipeline(LLM_PATH, "CPU");
    });

    it("generate with json schema in structured_output_config", async () => {
      const generationConfig = {
        max_new_tokens: 50,
        structured_output_config: {
          json_schema: JSON.stringify(person),
        },
      };
      const prompt = "Generate a json about a person.";
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
          structural_tags_config: StructuredOutputConfig.JSONSchema(JSON.stringify(person)),
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
          grammar: `root ::= date
date ::= year "-" month "-" day
year ::= digit digit digit digit
month ::= digit digit
day ::= digit digit
digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"`,
        },
      };
      const prompt = "Generate a date of first day of 21st century";
      const res = await pipeline.generate(prompt, generationConfig);
      const text = res.texts[0].trim();
      assert.ok(/^(\d{4})-(\d{2})-(\d{2})$/.test(text), `Unexpected format: ${text}`);
    });

    it("generate with StructuredOutputConfig.EBNF in structured_output_config", async () => {
      const generationConfig = {
        max_new_tokens: 50,
        structured_output_config: {
          structural_tags_config: StructuredOutputConfig.EBNF(`root ::= date
date ::= year "-" month "-" day
year ::= digit digit digit digit
month ::= digit digit
day ::= digit digit
digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"`),
        },
      };
      const prompt = "Generate a date of first day of 21st century";
      const res = await pipeline.generate(prompt, generationConfig);
      const text = res.texts[0].trim();
      assert.ok(/^(\d{4})-(\d{2})-(\d{2})$/.test(text), `Unexpected format: ${text}`);
    });

    it("generate with StructuredOutputConfig.Concat in structured_output_config", async () => {
      const generationConfig = {
        max_new_tokens: 50,
        structured_output_config: {
          structural_tags_config: StructuredOutputConfig.Concat(
            StructuredOutputConfig.JSONSchema(JSON.stringify(person)),
            StructuredOutputConfig.Union(
              StructuredOutputConfig.Regex("a"),
              StructuredOutputConfig.Regex("b"),
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
      assert.ok(postfix === "a" || postfix === "b", `Unexpected postfix: ${postfix}`);

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
