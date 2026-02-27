// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Parser } from "./parsers.js";

export enum StreamingStatus {
  RUNNING,
  STOP,
  CANCEL,
}

/** controls the stopping condition for grouped beam search. The following values are possible?:
 - "EARLY" stops as soon as there are `num_beams` complete candidates.
 - "HEURISTIC" stops when is it unlikely to find better candidates.
 - "NEVER" stops when there cannot be better candidates.
 */
export enum StopCriteria {
  EARLY,
  HEURISTIC,
  NEVER,
}

export declare namespace StructuredOutputConfig {
  export type StructuralTag =
    | string
    | Regex
    | JSONSchema
    | EBNF
    | ConstString
    | AnyText
    | QwenXMLParametersFormat
    | Concat
    | Union
    | Tag
    | TriggeredTags
    | TagsWithSeparator;
  /** Regex structural tag constrains output using a regular expression. */
  export type Regex = {
    structuralTagType: "Regex";
    value: string;
  };
  /** JSONSchema structural tag constrains output to a JSON document that
   * must conform to the provided JSON Schema string. */
  export type JSONSchema = {
    structuralTagType: "JSONSchema";
    value: string;
  };
  /** EBNF structural tag constrains output using an EBNF grammar. */
  export type EBNF = {
    structuralTagType: "EBNF";
    value: string;
  };
  /** ConstString structural tag forces the generator to produce exactly
   * the provided constant string value. */
  export type ConstString = {
    structuralTagType: "ConstString";
    value: string;
  };
  /** AnyText structural tag allows any text for the portion
   * of output covered by this tag. */
  export type AnyText = {
    structuralTagType: "AnyText";
  };
  /** QwenXMLParametersFormat instructs the generator to output an XML
   * parameters block derived from the provided JSON schema. This is a
   * specialized helper for Qwen-style XML parameter formatting. */
  export type QwenXMLParametersFormat = {
    structuralTagType: "QwenXMLParametersFormat";
    jsonSchema: string;
  };
  /** Concat composes multiple structural tags in sequence. Each element
   * must be produced in the given order.
   *
   * Example: Concat(ConstString("a"), ConstString("b")) produces "ab".*/
  export type Concat = {
    structuralTagType: "Concat";
    elements: StructuralTag[];
  };
  /** Union composes multiple structural tags as alternatives. The
   * model may produce any one of the provided elements. */
  export type Union = {
    structuralTagType: "Union";
    elements: StructuralTag[];
  };
  /** Tag defines a begin/end wrapper with constrained inner content.
   *
   * The generator will output `begin`, then the `content` (a StructuralTag),
   * and finally `end`.
   *
   * Example: Tag("<think>", AnyText(), "</think>") represents thinking portion of the model output. */
  export type Tag = {
    structuralTagType: "Tag";
    begin: string;
    content: StructuralTag;
    end: string;
  };
  /** TriggeredTags associates a set of `triggers` with multiple `tags`.
   *
   * When the model generates any of the trigger strings the structured generation
   * activates to produce configured tags. Flags allow requiring
   * at least one tag and stopping structured generation after the first tag. */
  export type TriggeredTags = {
    structuralTagType: "TriggeredTags";
    triggers: string[];
    tags: Tag[];
    atLeastOne: boolean;
    stopAfterFirst: boolean;
  };
  /** TagsWithSeparator configures generation of a sequence of tags
   *        separated by a fixed `separator` string.
   *
   * Can be used to produce repeated tagged elements like
   * "<f>A</f>;<f>B</f>" where `separator`=";". */
  export type TagsWithSeparator = {
    structuralTagType: "TagsWithSeparator";
    tags: Tag[];
    separator: string;
    atLeastOne: boolean;
    stopAfterFirst: boolean;
  };
}

/** This object is used to store the configuration for structured generation, which includes
 * the JSON schema and other related parameters. */
export class StructuredOutputConfig {
  /** if set, the output will be a JSON string constraint by the specified json-schema. */
  json_schema?: string;
  /** if set, the output will be constraint by specified regex.*/
  regex?: string;
  /** if set, the output will be constraint by specified EBNF grammar. */
  grammar?: string;
  /** if set, the output will be constraint by specified structural tags configuration. */
  structural_tags_config?: StructuredOutputConfig.StructuralTag;

  constructor(params: {
    json_schema?: string;
    regex?: string;
    grammar?: string;
    structural_tags_config?: StructuredOutputConfig.StructuralTag;
  }) {
    const { json_schema, regex, grammar, structural_tags_config } = params;
    this.json_schema = json_schema;
    this.regex = regex;
    this.grammar = grammar;
    this.structural_tags_config = structural_tags_config;
  }

  static JSONSchema(value: string): StructuredOutputConfig.JSONSchema {
    return { structuralTagType: "JSONSchema", value };
  }

  static Regex(value: string): StructuredOutputConfig.Regex {
    return { structuralTagType: "Regex", value };
  }

  static EBNF(value: string): StructuredOutputConfig.EBNF {
    return { structuralTagType: "EBNF", value };
  }

  static ConstString(value: string): StructuredOutputConfig.ConstString {
    return { structuralTagType: "ConstString", value };
  }

  static AnyText(): StructuredOutputConfig.AnyText {
    return { structuralTagType: "AnyText" };
  }

  static QwenXMLParametersFormat(
    jsonSchema: string,
  ): StructuredOutputConfig.QwenXMLParametersFormat {
    return { structuralTagType: "QwenXMLParametersFormat", jsonSchema };
  }

  static Concat(
    ...elements: StructuredOutputConfig.StructuralTag[]
  ): StructuredOutputConfig.Concat {
    return { structuralTagType: "Concat", elements };
  }

  static Union(...elements: StructuredOutputConfig.StructuralTag[]): StructuredOutputConfig.Union {
    return { structuralTagType: "Union", elements };
  }

  static Tag(params: {
    begin: string;
    content: StructuredOutputConfig.StructuralTag;
    end: string;
  }): StructuredOutputConfig.Tag {
    return { structuralTagType: "Tag", ...params };
  }

  static TriggeredTags(params: {
    triggers: string[];
    tags: StructuredOutputConfig.Tag[];
    atLeastOne?: boolean;
    stopAfterFirst?: boolean;
  }): StructuredOutputConfig.TriggeredTags {
    return {
      structuralTagType: "TriggeredTags",
      ...params,
      atLeastOne: params.atLeastOne ?? false,
      stopAfterFirst: params.stopAfterFirst ?? false,
    };
  }

  static TagsWithSeparator(params: {
    tags: StructuredOutputConfig.Tag[];
    separator: string;
    atLeastOne?: boolean;
    stopAfterFirst?: boolean;
  }): StructuredOutputConfig.TagsWithSeparator {
    return {
      structuralTagType: "TagsWithSeparator",
      ...params,
      atLeastOne: params.atLeastOne ?? false,
      stopAfterFirst: params.stopAfterFirst ?? false,
    };
  }
}

export type BeamSearchGenerationConfig = {
  /** number of beams for beam search. 1 disables beam search.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_beams?: number | bigint;
  /** number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_beam_groups?: number | bigint;
  /** value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time. */
  diversity_penalty?: number;
  /** exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
   * the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
   * likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
   * length_penalty < 0.0 encourages shorter sequences. */
  length_penalty?: number;
  /** the number of sequences to return for grouped beam search decoding.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_return_sequences?: number | bigint;
  /** if set to int > 0, all ngrams of that size can only occur once.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  no_repeat_ngram_size?: number | bigint;
  /** controls the stopping condition for grouped beam search. It accepts the following values?:
    - "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
    - "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
    - "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm). */
  stop_criteria?: StopCriteria;
};

export type RandomSamplingsGenerationConfig = {
  /** whether or not to use multinomial random sampling that add up to `top_p` or higher are kept. */
  do_sample?: boolean;
  /** the value used to modulate token probabilities for random sampling. */
  temperature?: number;
  /** the number of highest probability vocabulary tokens to keep for top-k-filtering.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  top_k?: number | bigint;
  /** if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. */
  top_p?: number;
  /** initializes random generator.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  rng_seed?: number | bigint;
};

export type AssistingGenerationConfig = {
  /**  the lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number update. */
  assistant_confidence_threshold?: number;
  /** the defined candidates number to be generated by draft model/prompt lookup in case of static strategy candidates number update.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_assistant_tokens?: number | bigint;
  /** is maximum ngram to use when looking for matches in the prompt.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_ngram_size?: number | bigint;
  /** whether to apply chat_template for non-chat scenarios */
  apply_chat_template?: boolean;
};

export type CDPrunerGenerationConfig = {
  /** the percentage of visual tokens to prune [0-100). Set to 0 to disable pruning. */
  pruning_ratio?: number;
  /** the weight of relevance for visual tokens. */
  relevance_weight?: number;
};

export type GenericGenerationConfig = {
  // adapters?: AdapterConfig | None
  /** if set to true, the model will echo the prompt in the output. */
  echo?: boolean;
  /** token_id of <eos> (end of sentence).
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  eos_token_id?: number | bigint;
  /** reduces absolute log prob as many times as the token was generated. */
  frequency_penalty?: number;
  /** if set to true, then generation will not stop even if <eos> token is met. */
  ignore_eos?: boolean;
  /** if set to true stop string that matched generation will be included in generation output (default?: false) */
  include_stop_str_in_output?: boolean;
  /** number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
   * Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default?: 0).
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  logprobs?: number | bigint;
  /** Maximum length the generated tokens can have. Corresponds to the length of the input prompt max_new_tokens.
   * Its effect is overridden by `max_new_tokens`, if also set.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_length?: number | bigint;
  /** Maximum numbers of tokens to generate, excluding the number of tokens in the prompt.
   * `max_new_tokens` has priority over `max_length`.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_new_tokens?: number | bigint;
  /** set 0 probability for eos_token_id for the first eos_token_id generated tokens.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  min_new_tokens?: number | bigint;
  /** reduces absolute log prob if the token was generated at least once. */
  presence_penalty?: number;
  /** the parameter for repetition penalty. 1.0 means no penalty. */
  repetition_penalty?: number;
  /** a set of strings that will cause pipeline to stop generating further tokens. */
  stop_strings?: Set<string>;
  /** a set of tokens that will cause pipeline to stop generating further tokens. */
  stop_token_ids?: Set<number | bigint>;
};

export type StructuredOutputGenerationConfig = {
  /** This object is used to store the configuration for structured generation, which includes
   * the JSON schema and other related parameters. */
  structured_output_config?: StructuredOutputConfig;
};

export type ParserGenerationConfig = {
  /** Array of parsers to process complete text content at the end of generation */
  parsers?: Parser[];
};

export type GenerationConfig = GenericGenerationConfig &
  BeamSearchGenerationConfig &
  RandomSamplingsGenerationConfig &
  CDPrunerGenerationConfig &
  AssistingGenerationConfig &
  StructuredOutputGenerationConfig &
  ParserGenerationConfig;

export type SchedulerConfig = {
  /** a maximum number of tokens to batch
   * (in contrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
   * When ContinuousBatching is invoked from LLMPipeline (client scenario) by default max_num_batched_tokens is not limited.
   * Default: 256
   */
  max_num_batched_tokens?: number;
  /** total number of KV blocks available to scheduler logic
   * Default: 0
   */
  num_kv_blocks?: number;
  /** total size of KV cache in GB
   * When both num_kv_blocks and cache_size are set, num_kv_blocks is used.
   * When both num_kv_blocks and cache_size are equal to zero dynamic KV-cache allocation is turned on.
   * Default: 0
   */
  cache_size?: number;
  /** whether to split prompt / generate to different scheduling phases
   * Allows to process prompt partially in case when batch size is limited.
   * If dynamic_split_fuse is turned off any prompt that is longer than batch size will lead to error.
   * Default: true
   */
  dynamic_split_fuse?: boolean;
};

export type LLMPipelineProperties = {
  schedulerConfig?: SchedulerConfig;
} & Record<string, unknown>;

export type VLMPipelineProperties = {
  schedulerConfig?: SchedulerConfig;
} & Record<string, unknown>;
