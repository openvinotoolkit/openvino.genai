// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { ChatHistory } from "../dist/index.js";

describe("ChatHistory", () => {
  let history;

  beforeEach(() => {
    history = new ChatHistory();
  });

  describe("Constructor", () => {
    it("should create an empty chat history", () => {
      assert.strictEqual(history.size(), 0);
      assert.strictEqual(history.empty(), true);
    });

    it("should create chat history with initial messages", () => {
      const messages = [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Hello!" },
      ];
      const historyWithMessages = new ChatHistory(messages);

      assert.strictEqual(historyWithMessages.size(), 2);
      assert.strictEqual(historyWithMessages.empty(), false);
    });
  });

  describe("push", () => {
    it("should add a message to history", () => {
      history.push({ role: "user", content: "Test message" });

      assert.strictEqual(history.size(), 1);
      assert.strictEqual(history.empty(), false);
    });

    it("should support method chaining", () => {
      const result = history
        .push({ role: "system", content: "System message" })
        .push({ role: "user", content: "User message" });

      assert.strictEqual(result, history);
      assert.strictEqual(history.size(), 2);
    });

    it("should accept messages with additional properties", () => {
      history.push({
        role: "assistant",
        content: "Response",
        metadata: { confidence: 0.95 },
      });

      assert.strictEqual(history.size(), 1);
    });
  });

  describe("pop", () => {
    it("should remove the last message", () => {
      history.push({ role: "user", content: "Message 1" });
      history.push({ role: "user", content: "Message 2" });

      assert.strictEqual(history.size(), 2);

      history.pop();

      assert.strictEqual(history.size(), 1);
    });

    it("should throw when popping from empty history", () => {
      assert.throws(() => {
        history.pop();
      });
    });
  });

  describe("messages", () => {
    it("should return all messages in order", () => {
      const messages = [
        { role: "system", content: "System message" },
        { role: "user", content: "User message" },
        { role: "assistant", content: "Assistant message" },
      ];

      messages.forEach((msg) => history.push(msg));

      const retrievedMessages = history.getMessages();

      assert.deepStrictEqual(retrievedMessages, messages);
    });

    it("setMessages should replace all messages", () => {
      history.push({ role: "user", content: "Old message" });

      const newMessages = [
        { role: "user", content: "Initial message 1" },
        { role: "user", content: "Initial message 2" },
      ];
      history.setMessages(newMessages);

      assert.deepStrictEqual(history.getMessages(), newMessages);
    });

    it("throws when setting invalid messages", () => {
      assert.throws(() => {
        history.setMessages("invalid messages");
      });
    });
  });

  describe("size and empty", () => {
    it("should return correct size", () => {
      assert.strictEqual(history.size(), 0);

      history.push({ role: "user", content: "1" });
      assert.strictEqual(history.size(), 1);

      history.push({ role: "user", content: "2" });
      assert.strictEqual(history.size(), 2);
    });

    it("should correctly report empty status", () => {
      assert.strictEqual(history.empty(), true);

      history.push({ role: "user", content: "Test" });
      assert.strictEqual(history.empty(), false);

      history.pop();
      assert.strictEqual(history.empty(), true);
    });
  });

  describe("clear", () => {
    it("should remove all messages", () => {
      history
        .push({ role: "user", content: "1" })
        .push({ role: "user", content: "2" })
        .push({ role: "user", content: "3" });

      assert.strictEqual(history.size(), 3);

      history.clear();

      assert.strictEqual(history.size(), 0);
      assert.strictEqual(history.empty(), true);
    });
  });

  describe("tools", () => {
    it("should set tools array", () => {
      const tools = [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get weather",
            parameters: { type: "object" },
          },
        },
      ];

      const result = history.setTools(tools).getTools();

      assert.deepStrictEqual(result, tools);
    });

    it("throws when setting invalid tools", () => {
      assert.throws(() => {
        history.setTools("invalid tool");
      });
    });
  });

  describe("setExtraContext", () => {
    it("should set extra context", () => {
      const context = {
        userName: "Alice",
        sessionId: "123",
      };

      history.setExtraContext(context);
      const result = history.getExtraContext();

      assert.deepStrictEqual(result, context);
    });

    it("should support method chaining", () => {
      const result = history
        .push({ role: "user", content: "Test" })
        .setExtraContext({ key: "value" });

      assert.strictEqual(result, history);
    });
  });
});
