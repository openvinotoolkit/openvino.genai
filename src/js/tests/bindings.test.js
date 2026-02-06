import { LLMPipeline } from "../dist/addon.js";

import assert from "node:assert";
import { describe, it, before } from "node:test";

const { LLM_PATH } = process.env;

if (!LLM_PATH) {
  throw new Error("Please set LLM_PATH environment variable to run the tests.");
}

describe("bindings", () => {
  let pipeline = null;

  before((_, done) => {
    pipeline = new LLMPipeline();

    pipeline.init(LLM_PATH, "CPU", {}, (err) => {
      if (err) {
        console.error(err);
        process.exit(1);
      }
      done();
    });
  });

  it("should generate string result", (_, done) => {
    let output = "";

    pipeline.generate(
      "Continue: 1 2 3",
      { temperature: 0, max_new_tokens: 4 },
      (chunk) => {
        output += chunk;
      },
      (error, result) => {
        assert.strictEqual(error, null);
        assert.ok(output.length > 0);
        assert.strictEqual(output, result.texts[0]);
        done();
      },
    );
  });
});
