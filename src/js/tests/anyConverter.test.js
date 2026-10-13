// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const addonPath = [
  path.join(__dirname, "converter_ext", "build", "converter_test_ext.node"),
  path.join(__dirname, "converter_ext", "converter_test_ext.node"),
].find((p) => fs.existsSync(p));

if (process.env.REQUIRE_CONVERTER_EXT === "1" && !addonPath) {
  throw new Error("Converter test addon is required but was not built");
}

const ext = addonPath ? require(addonPath) : null;
const suite = addonPath ? describe : describe.skip;

suite("ov::Any -> JS native converter", () => {
  it("string -> string", () => {
    const v = ext.convString();
    assert.strictEqual(typeof v, "string");
    assert.strictEqual(v, "HAPPY");
  });

  it("const char* is stored as std::string", () => {
    assert.strictEqual(ext.cstrStoredAsString(), true);
  });

  it("bool -> boolean (not number)", () => {
    const v = ext.convBool();
    assert.strictEqual(typeof v, "boolean");
    assert.strictEqual(v, true);
  });

  it("int -> number", () => {
    const v = ext.convInt();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, -42);
  });

  it("safe-range int64 -> number", () => {
    const v = ext.convInt64Safe();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, 42);
  });

  it("out-of-range int64 -> BigInt", () => {
    const v = ext.convInt64Big();
    assert.strictEqual(typeof v, "bigint");
    assert.strictEqual(v, 9007199254740993n);
  });

  it("safe-range size_t -> number", () => {
    const v = ext.convSizeT();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, 7);
  });

  it("out-of-range size_t -> BigInt", () => {
    const v = ext.convSizeTBig();
    assert.strictEqual(typeof v, "bigint");
    assert.strictEqual(v, 9007199254740993n);
  });

  it("int64 +(2^53 - 1) boundary -> number", () => {
    const v = ext.convInt64MaxSafe();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, 9007199254740991);
  });

  it("int64 +2^53 (just outside) -> BigInt", () => {
    const v = ext.convInt64JustAbove();
    assert.strictEqual(typeof v, "bigint");
    assert.strictEqual(v, 9007199254740992n);
  });

  it("int64 -(2^53 - 1) boundary -> number", () => {
    const v = ext.convInt64MinSafe();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, -9007199254740991);
  });

  it("int64 -2^53 (just outside) -> BigInt", () => {
    const v = ext.convInt64JustBelow();
    assert.strictEqual(typeof v, "bigint");
    assert.strictEqual(v, -9007199254740992n);
  });

  it("size_t (2^53 - 1) boundary -> number", () => {
    const v = ext.convSizeTMaxSafe();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, 9007199254740991);
  });

  it("size_t 2^53 (just outside) -> BigInt", () => {
    const v = ext.convSizeTJustAbove();
    assert.strictEqual(typeof v, "bigint");
    assert.strictEqual(v, 9007199254740992n);
  });

  it("float -> number (rounded convention)", () => {
    const v = ext.convFloat();
    assert.strictEqual(typeof v, "number");
    assert.ok(Math.abs(v - 1.5) < 1e-6);
  });

  it("double -> number", () => {
    const v = ext.convDouble();
    assert.strictEqual(typeof v, "number");
    assert.ok(Math.abs(v - 2.5) < 1e-12);
  });

  it("nested AnyMap -> nested object", () => {
    const v = ext.convNestedMap();
    assert.deepStrictEqual(v, { nested: { emotion: "HAPPY" }, flag: true });
  });

  it("vector<string> -> string[]", () => {
    const v = ext.convVecString();
    assert.deepStrictEqual(v, ["a", "b"]);
  });

  it("vector<int64_t> -> number[]", () => {
    const v = ext.convVecInt64();
    assert.deepStrictEqual(v, [1, 2, 3]);
  });

  it("vector<double> -> number[]", () => {
    const v = ext.convVecDouble();
    assert.ok(v.every((x) => typeof x === "number"));
    assert.ok(Math.abs(v[0] - 1.5) < 1e-12 && Math.abs(v[1] - 2.5) < 1e-12);
  });

  it("vector<float> -> number[]", () => {
    const v = ext.convVecFloat();
    assert.ok(v.every((x) => typeof x === "number"));
    assert.ok(Math.abs(v[0] - 1.5) < 1e-6 && Math.abs(v[1] - 2.5) < 1e-6);
  });

  it("vector<ov::Any> recurses per element", () => {
    const v = ext.convVecAny();
    assert.strictEqual(v.length, 3);
    assert.strictEqual(v[0], "x");
    assert.strictEqual(typeof v[1], "number");
    assert.strictEqual(v[1], 5);
    assert.strictEqual(v[2], true);
  });

  it("empty ov::Any -> null", () => {
    assert.strictEqual(ext.convEmptyAny(), null);
  });

  it("empty AnyMap -> {}", () => {
    const v = ext.convEmptyMap();
    assert.strictEqual(typeof v, "object");
    assert.deepStrictEqual(v, {});
  });

  it("SenseVoice-shaped features", () => {
    const v = ext.convFeaturesExample();
    assert.deepStrictEqual(v, [{ emotion: "HAPPY", event: "Speech" }]);
  });

  it("unsupported type throws", () => {
    assert.throws(() => ext.convUnsupported());
  });
});
