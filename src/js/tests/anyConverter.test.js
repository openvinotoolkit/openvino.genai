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
  it("SenseVoice-shaped features", () => {
    const v = ext.convFeaturesExample();
    assert.deepStrictEqual(v, [{ emotion: "HAPPY", event: "Speech" }]);
  });

  it("nested AnyMap -> nested object", () => {
    const v = ext.convNestedMap();
    assert.deepStrictEqual(v, { nested: { emotion: "HAPPY" }, flag: true });
  });

  it("vector<ov::Any> recurses per element", () => {
    const v = ext.convVecAny();
    assert.strictEqual(v.length, 3);
    assert.strictEqual(v[0], "x");
    assert.strictEqual(typeof v[1], "number");
    assert.strictEqual(v[1], 5);
    assert.strictEqual(v[2], true);
  });

  it("unsupported type throws", () => {
    assert.throws(
      () => ext.convUnsupported(),
      /unsupported stored type/,
    );
  });

  it("empty AnyMap -> {}", () => {
    const v = ext.convEmptyMap();
    assert.strictEqual(typeof v, "object");
    assert.deepStrictEqual(v, {});
  });

  it("bool -> boolean (not number)", () => {
    const v = ext.convBool();
    assert.strictEqual(typeof v, "boolean");
    assert.strictEqual(v, true);
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

  it("empty ov::Any -> null", () => {
    assert.strictEqual(ext.convEmptyAny(), null);
  });

  it("float -> number (rounded convention)", () => {
    const v = ext.convFloat();
    assert.strictEqual(typeof v, "number");
    assert.strictEqual(v, 1.2345679);
  });
});
