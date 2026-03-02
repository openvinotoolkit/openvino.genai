// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { LLMPipeline, StopCriteria } from "../dist/index.js";

import assert from "node:assert/strict";
import { describe, it, before, afterEach } from "node:test";

const { LLM_PATH } = process.env;

if (!LLM_PATH) {
  throw new Error("Please set LLM_PATH environment variable to run the tests.");
}

/** Compare two values that may be Set or array; for Sets compare sorted array form. */
function valuesEqual(a, b) {
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object") return a === b;
  if (a instanceof Set) {
    if (!(b instanceof Set)) return false;
    const arrA = [...a].sort();
    const arrB = [...b].sort();
    return arrA.length === arrB.length && arrA.every((v, i) => valuesEqual(v, arrB[i]));
  }
  if (Array.isArray(a)) {
    if (!Array.isArray(b)) return false;
    return a.length === b.length && a.every((v, i) => valuesEqual(v, b[i]));
  }
  return false;
}

describe("GenerationConfig JS <-> C++ conversion", () => {
  let pipeline = null;
  let initialConfig = null;

  /** Assert that a subset of config fields round-trip correctly. */
  function assertRoundTrip(configPatch) {
    pipeline.setGenerationConfig(configPatch);
    const back = pipeline.getGenerationConfig();

    for (const key of Object.keys(configPatch)) {
      const expected = configPatch[key];
      const actual = back[key];
      assert.ok(
        valuesEqual(expected, actual),
        `round-trip mismatch for ${key}: expected ${expected}, actual ${actual}`,
      );
    }
  }

  before(async () => {
    pipeline = await LLMPipeline(LLM_PATH, "CPU");
    initialConfig = pipeline.getGenerationConfig();
  });

  afterEach(() => pipeline.setGenerationConfig(initialConfig));

  describe("stop_criteria (numeric enum)", () => {
    it("round-trips StopCriteria.EARLY (0)", () => {
      assertRoundTrip({ stop_criteria: StopCriteria.EARLY });
    });

    it("round-trips StopCriteria.HEURISTIC (1)", () => {
      assertRoundTrip({ stop_criteria: StopCriteria.HEURISTIC });
    });

    it("round-trips StopCriteria.NEVER (2)", () => {
      assertRoundTrip({ stop_criteria: StopCriteria.NEVER });
    });

    it("throws when stop_criteria is out of range", () => {
      assert.throws(
        () => pipeline.setGenerationConfig({ stop_criteria: 3 }),
        /Invalid stop criteria/,
      );
    });

    it("throws when stop_criteria is not a number", () => {
      assert.throws(
        () => pipeline.setGenerationConfig({ stop_criteria: "EARLY" }),
        /stop_criteria must be a number/,
      );
    });
  });

  describe("numeric and boolean fields", () => {
    it("round-trips max_new_tokens, temperature, do_sample", () => {
      assertRoundTrip({
        max_new_tokens: 42,
        temperature: 0.7,
        do_sample: true,
      });
    });

    it("round-trips top_p, top_k, repetition_penalty", () => {
      assertRoundTrip({
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
      });
    });

    it("round-trips echo, ignore_eos, include_stop_str_in_output", () => {
      assertRoundTrip({
        echo: true,
        ignore_eos: false,
        include_stop_str_in_output: true,
      });
    });

    it("round-trips beam search fields", () => {
      assertRoundTrip({
        num_beams: 2,
        num_beam_groups: 2,
        length_penalty: 1.0,
        diversity_penalty: 0.5,
        num_return_sequences: 1,
        repetition_penalty: 1.0,
        do_sample: false,
      });
    });
  });

  describe("stop_strings and stop_token_ids (Set)", () => {
    it("round-trips stop_strings as Set", () => {
      assertRoundTrip({
        stop_strings: new Set(["end", "stop", ...(initialConfig.stop_strings ?? [])]),
      });
    });

    it("round-trips stop_token_ids as Set", () => {
      const stopTokenIds = new Set([1, 2, 3, ...(initialConfig.stop_token_ids ?? [])]);
      assertRoundTrip({ stop_token_ids: stopTokenIds });
    });
  });

  describe("check types", () => {
    it("check bigint and number", () => {
      const config = pipeline.getGenerationConfig();
      assert.strictEqual(typeof config.max_new_tokens, "bigint");
      pipeline.setGenerationConfig({ max_new_tokens: 100 });
      const updatedConfig = pipeline.getGenerationConfig();
      assert.strictEqual(typeof updatedConfig.max_new_tokens, "number");
    });

    it("throws if too large bigint value used", () => {
      assert.throws(
        () => pipeline.setGenerationConfig({ max_new_tokens: BigInt(2 ** 100) }),
        /BigInt value is too large/,
      );
    });
  });
});
