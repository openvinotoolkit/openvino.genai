/* eslint-disable @typescript-eslint/no-misused-new */
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Tensor } from "openvino-node";
import type { ChatHistory } from "./chatHistory.js";

/**
 * TokenizedInputs contains input_ids, attention_mask and (optionally) token_type_ids tensors.
 * token_type_ids is returned if the tokenizer supports paired input, otherwise the field is undefined.
 * This is the result of encoding prompts using the Tokenizer.
 */
export interface TokenizedInputs {
  /** Tensor containing token IDs of the encoded input */
  input_ids: Tensor;
  /** Tensor containing attention mask (1 for real tokens, 0 for padding) */
  attention_mask: Tensor;
  /**
   * Optional tensor with token type IDs (segment ids) for paired inputs.
   * Present only if the model/tokenizer supports paired input.
   */
  token_type_ids?: Tensor;
}

/**
 * Options for encode method.
 */
export interface EncodeOptions {
  /**
   * Whether to add special tokens like BOS, EOS, PAD.
   * @defaultValue true
   */
  add_special_tokens?: boolean;

  /**
   * Whether to pad the sequence to the maximum length.
   * @defaultValue false
   */
  pad_to_max_length?: boolean;

  /**
   * Maximum length of the sequence.
   * If undefined, the value will be taken from the IR.
   */
  max_length?: number;

  /**
   * Side to pad the sequence, can be 'left' or 'right'.
   * If undefined, the value will be taken from the IR.
   */
  padding_side?: "left" | "right";
}

/**
 * Options for decode method.
 */
export interface DecodeOptions {
  /**
   * Whether to skip special tokens like BOS, EOS, PAD during detokenization.
   * @defaultValue true
   */
  skip_special_tokens?: boolean;
}

/**
 * The Tokenizer class is used to encode prompts and decode resulting tokens.
 *
 * Chat template is initialized from sources in the following order, overriding the previous value:
 * 1. chat_template entry from tokenizer_config.json
 * 2. chat_template entry from processor_config.json
 * 3. chat_template entry from chat_template.json
 * 4. chat_template entry from rt_info section of openvino.Model
 * 5. If the template is known to be not supported by GenAI, it's replaced with a simplified supported version.
 */
export interface Tokenizer {
  /**
   * Load tokenizer and detokenizer IRs by path.
   * @param tokenizerPath Path to a directory containing tokenizer/detokenizer XML/BIN files.
   * @param properties Optional OpenVINO compilation properties.
   */
  new (tokenizerPath: string, properties?: Record<string, unknown>): Tokenizer;

  /**
   * Create tokenizer from already loaded IR contents.
   * @param tokenizerModel Tokenizer XML string.
   * @param tokenizerWeights Tokenizer weights tensor.
   * @param detokenizerModel Detokenizer XML string.
   * @param detokenizerWeights Detokenizer weights tensor.
   * @param properties Optional OpenVINO compilation properties.
   */
  new (
    tokenizerModel: string,
    tokenizerWeights: Tensor,
    detokenizerModel: string,
    detokenizerWeights: Tensor,
    properties?: Record<string, unknown>,
  ): Tokenizer;

  /**
   * Applies a chat template to format chat history into a prompt string.
   * @param chatHistory - chat history as an array of message objects or ChatHistory instance
   * @param addGenerationPrompt - whether to add a generation prompt at the end
   * @param chatTemplate - optional custom chat template to use instead of the default
   * @param tools - optional array of tool definitions for function calling
   * @param extraContext - optional extra context object for custom template variables
   * @returns formatted prompt string
   */
  applyChatTemplate(
    chatHistory: Record<string, unknown>[] | ChatHistory,
    addGenerationPrompt: boolean,
    chatTemplate?: string,
    tools?: Record<string, unknown>[],
    extraContext?: Record<string, unknown>,
  ): string;

  /**
   * Encodes a single prompt or a list of prompts into tokenized inputs.
   * @param prompts - single prompt string or array of prompts
   * @param options - encoding options
   * @returns TokenizedInputs object containing input_ids, attention_mask and optional token_type_ids tensors.
   */
  encode(prompts: string | string[], options?: EncodeOptions): TokenizedInputs;

  /**
   * Encodes two lists of prompts into tokenized inputs (for paired input).
   * The number of strings must be the same, or one of the inputs can contain one string.
   * In the latter case, the single-string input will be broadcast into the shape of the other input,
   * which is more efficient than repeating the string in pairs.
   * @param prompts1 - first list of prompts to encode
   * @param prompts2 - second list of prompts to encode
   * @param options - encoding options
   * @returns TokenizedInputs object containing input_ids, attention_mask and optional token_type_ids tensors.
   */
  encode(prompts1: string[], prompts2: string[], options?: EncodeOptions): TokenizedInputs;

  /**
   * Encodes a list of paired prompts into tokenized inputs.
   * Input format is same as for HF paired input [[prompt_1, prompt_2], ...].
   * @param prompts - list of paired prompts to encode
   * @param options - encoding options
   * @returns TokenizedInputs object containing input_ids, attention_mask and optional token_type_ids tensors.
   */
  encode(prompts: [string, string][], options?: EncodeOptions): TokenizedInputs;

  /**
   * Decode a sequence of token IDs into a string prompt.
   *
   * @param tokens - sequence of token IDs to decode
   * @param options - decoding options
   * @returns decoded string.
   */
  decode(tokens: number[] | bigint[], options?: DecodeOptions): string;

  /**
   * Decode a batch of token sequences (as Tensor or array of arrays) into a list of string prompts.
   *
   * @param tokens - tensor containing token IDs or batch of token ID sequences
   * @param options - decoding options
   * @returns list of decoded strings.
   */
  decode(tokens: Tensor | number[][] | bigint[][], options?: DecodeOptions): string[];

  /**
   * Returns the BOS (Beginning of Sequence) token string.
   * @returns BOS token string
   */
  getBosToken(): string;

  /**
   * Returns the BOS (Beginning of Sequence) token ID.
   * @returns BOS token ID
   */
  getBosTokenId(): bigint;

  /**
   * Returns the EOS (End of Sequence) token string.
   * @returns EOS token string
   */
  getEosToken(): string;

  /**
   * Returns the EOS (End of Sequence) token ID.
   * @returns EOS token ID
   */
  getEosTokenId(): bigint;

  /**
   * Returns the PAD (Padding) token string.
   * @returns PAD token string
   */
  getPadToken(): string;

  /**
   * Returns the PAD (Padding) token ID.
   * @returns PAD token ID
   */
  getPadTokenId(): bigint;

  /**
   * Returns the current chat template string.
   * @returns current chat template string
   */
  getChatTemplate(): string;

  /**
   * Returns the original chat template from the tokenizer configuration.
   * @returns original chat template string
   */
  getOriginalChatTemplate(): string;

  /**
   * Override a chat template read from tokenizer_config.json.
   * @param chatTemplate - custom chat template string to use
   */
  setChatTemplate(chatTemplate: string): void;

  /**
   * Returns true if the tokenizer supports paired input, false otherwise.
   * @returns whether the tokenizer supports paired input
   */
  supportsPairedInput(): boolean;

  /**
   * The current chat template string.
   * Can be used to get or set the chat template.
   */
  chatTemplate: string;
}
