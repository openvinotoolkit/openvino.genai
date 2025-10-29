/* eslint-disable @typescript-eslint/no-misused-new */
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Represents a single message in a chat conversation.
 */
export type ChatMessage = Record<string, any>;

/**
 * Represents a tool definition for function calling.
 */
export type ToolDefinition = Record<string, any>;

/**
 * Extra context for custom template variables.
 */
export type ExtraContext = Record<string, any>;

/**
 * ChatHistory manages conversation messages and optional metadata for chat templates.
 *
 * @example
 * ```typescript
 * // Create an empty chat history
 * const history = new ChatHistory();
 *
 * // Add messages
 * history.push({ role: "system", content: "You are a helpful assistant." });
 * history.push({ role: "user", content: "Hello!" });
 * history.push({ role: "assistant", content: "Hi! How can I help you?" });
 *
 * // Create with initial messages
 * const history2 = new ChatHistory([
 *   { role: "system", content: "You are a helpful assistant." },
 *   { role: "user", content: "What's the weather?" }
 * ]);
 *
 * // Set tools for function calling
 * history.setTools([
 *   {
 *     type: "function",
 *     function: {
 *       name: "get_weather",
 *       description: "Get the current weather",
 *       parameters: {
 *         type: "object",
 *         properties: {
 *           location: { type: "string" }
 *         }
 *       }
 *     }
 *   }
 * ]);
 * ```
 */
export interface ChatHistory {
  /**
   * Creates a new ChatHistory instance.
   * @param messages - Optional array of initial messages
   */
  new (messages?: ChatMessage[]): ChatHistory;

  /**
   * Adds a message to the end of the chat history.
   * @param message - The message to add
   * @returns The ChatHistory instance for method chaining
   *
   * @example
   * ```typescript
   * history.push({ role: "user", content: "Hello!" });
   * ```
   *
   * @note The message structure is flexible and depends on the model used.
   */
  push(message: ChatMessage): ChatHistory;
  /**
   * Removes the last message from the chat history.
   * @throws {Error} If the history is empty
   */
  pop(): void;
  /**
   * Returns all messages in the chat history.
   * @returns Array of messages
   */
  getMessages(): ChatMessage[];
  /**
   * Replace all messages with a new list.
   *
   * @param messages - The new list of messages
   * @returns The ChatHistory instance for method chaining
   */
  setMessages(messages: ChatMessage[]): ChatHistory;
  /**
   * Removes all messages from the chat history.
   */
  clear(): void;
  /**
   * Returns the number of messages in the chat history.
   * @returns The number of messages
   */
  size(): number;
  /**
   * Checks if the chat history is empty.
   * @returns true if the history contains no messages, false otherwise
   *
   * @example
   * ```typescript
   * if (history.empty()) {
   *   console.log("No messages yet");
   * }
   * ```
   */
  empty(): boolean;
  /**
   * Sets the tools array for function calling.
   * @param tools - Array of tool definitions
   * @returns The ChatHistory instance for method chaining
   *
   * @example
   * ```typescript
   * history.setTools([
   *   {
   *     type: "function",
   *     function: {
   *       name: "get_weather",
   *       description: "Get current weather",
   *       parameters: {
   *         type: "object",
   *         properties: {
   *           location: { type: "string", description: "City name" }
   *         },
   *         required: ["location"]
   *       }
   *     }
   *   }
   * ]);
   * ```
   *
   * @note The tool structure is flexible and depends on the model used.
   */
  setTools(tools: ToolDefinition[]): ChatHistory;
  /**
   * Gets the tools array.
   * @returns The tools array
   */
  getTools(): ToolDefinition[];
  /**
   * Sets extra context for custom template variables.
   * @param context - Object containing custom variables
   * @returns The ChatHistory instance for method chaining
   *
   * @example
   * ```typescript
   * history.setExtraContext({
   *   user_name: "Alice",
   *   date: "2025-01-01",
   *   custom_instruction: "Be concise"
   * });
   * ```
   *
   * @note The extra context structure is flexible and depends on the model used.
   */
  setExtraContext(context: ExtraContext): ChatHistory;
  /**
   * Gets the extra context.
   * @returns The extra context object
   */
  getExtraContext(): ExtraContext;
}
