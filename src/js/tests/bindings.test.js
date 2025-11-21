import addon from "../dist/addon.js";

import assert from "node:assert";
import { describe, it, before, after } from "node:test";
import { models } from "./models.js";

const MODEL_PATH = process.env.MODEL_PATH || `./tests/models/${models.LLM.split("/")[1]}`;

describe("bindings", () => {
  let pipeline = null;

  before((_, done) => {
    pipeline = new addon.LLMPipeline();

    pipeline.init(MODEL_PATH, "CPU", {}, (err) => {
      if (err) {
        console.error(err);
        process.exit(1);
      }

      pipeline.startChat("", (err) => {
        if (err) {
          console.error(err);
          process.exit(1);
        }

        done();
      });
    });
  });

  after((_, done) => {
    pipeline.finishChat((err) => {
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
      (isDone, chunk) => {
        if (!isDone) {
          output += chunk;

          return;
        }

        assert.ok(output.length > 0);
        done();
      },
      { temperature: "0", max_new_tokens: "4" },
    );
  });
});
