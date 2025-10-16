// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
  export type CompoundGrammar = Regex | JSONSchema | EBNF | Concat | Union;
  export type Regex = {
    regex: string;
  };
  export type JSONSchema = {
    json_schema: string;
  };
  export type EBNF = {
    grammar?: string;
  };
  export type Concat = {
    compoundType: "Concat";
    left: CompoundGrammar;
    right: CompoundGrammar;
  };
  export type Union = {
    compoundType: "Union";
    left: CompoundGrammar;
    right: CompoundGrammar;
  };
}

/** Structure to keep generation config parameters for structural tags in structured output generation.
 * It is used to store the configuration for a single structural tag item, which includes the begin string,
 * schema, and end string. */
export type StructuralTagItem = {
  /** the string that marks the beginning of the structural tag. */
  begin: string;
  /** the JSON schema that defines the structure of the tag. */
  schema: string;
  /** the string that marks the end of the structural tag. */
  end: string;
};

/** Configures structured output generation by combining regular sampling with structural tags.
 *
 * When the model generates a trigger string, it switches to structured output mode and produces output
 * based on the defined structural tags. Afterward, regular sampling resumes.
 *
 * Example:
 *   - Trigger "<func=" activates tags with begin "<func=sum>" or "<func=multiply>".
 *
 * Note:
 *   - Simple triggers like "<" may activate structured output unexpectedly if present in regular text.
 *   - Very specific or long triggers may be difficult for the model to generate,
 *     so structured output may not be triggered. */
export type StructuralTagsConfig = {
  /** List of StructuralTagItem objects defining structural tags. */
  structural_tags: StructuralTagItem[];
  /** List of strings that trigger structured output generation.
   * Triggers may match the beginning or part of a tag's begin string. */
  triggers: string[];
};

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
  structural_tags_config?: StructuralTagsConfig;
  /** if set, the output will be constraint by specified compound grammar.
   * Compound grammar is a combination of multiple grammars that can be used to generate structured outputs.
   * It allows for more complex and flexible structured output generation.
   * The compound grammar a Union or Concat of several grammars, where each grammar can be a JSON schema, regex, EBNF, Union or Concat. */
  compound_grammar?: StructuredOutputConfig.CompoundGrammar;

  constructor(params: {
    json_schema?: string;
    regex?: string;
    grammar?: string;
    structural_tags_config?: StructuralTagsConfig;
    compound_grammar?: StructuredOutputConfig.CompoundGrammar;
  }) {
    const { json_schema, regex, grammar, structural_tags_config, compound_grammar } = params;
    this.json_schema = json_schema;
    this.regex = regex;
    this.grammar = grammar;
    this.structural_tags_config = structural_tags_config;
    this.compound_grammar = compound_grammar;
  }

  /** JSON schema building block for compound grammar configuration. */
  static JSONSchema(json_schema: string): StructuredOutputConfig.JSONSchema {
    return { json_schema };
  }
  /** Regex building block for compound grammar configuration. */
  static Regex(regex: string): StructuredOutputConfig.Regex {
    return { regex: regex };
  }
  /** EBNF grammar building block for compound grammar configuration. */
  static EBNF(grammar?: string): StructuredOutputConfig.EBNF {
    return { grammar: grammar };
  }
  /** Concat combines two grammars sequentially, e.g. "A B" means A followed by B */
  static Concat(
    left: StructuredOutputConfig.CompoundGrammar,
    right: StructuredOutputConfig.CompoundGrammar,
  ): StructuredOutputConfig.Concat {
    return { compoundType: "Concat", left, right };
  }
  /** Union combines two grammars in parallel, e.g. "A | B" means either A or B */
  static Union(
    left: StructuredOutputConfig.CompoundGrammar,
    right: StructuredOutputConfig.CompoundGrammar,
  ): StructuredOutputConfig.Union {
    return { compoundType: "Union", left, right };
  }
}

export type BeamSearchGenerationConfig = {
  /** number of beams for beam search. 1 disables beam search. */
  num_beams?: number;
  /** number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. */
  num_beam_groups?: number;
  /** value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time. */
  diversity_penalty?: number;
  /** exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
   * the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
   * likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
   * length_penalty < 0.0 encourages shorter sequences. */
  length_penalty?: number;
  /** the number of sequences to return for grouped beam search decoding. */
  num_return_sequences?: number;
  /** if set to int > 0, all ngrams of that size can only occur once. */
  no_repeat_ngram_size?: number;
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
  /** the number of highest probability vocabulary tokens to keep for top-k-filtering. */
  top_k?: number;
  /** if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. */
  top_p?: number;
  /** initializes random generator. */
  rng_seed?: number;
  /** the number of sequences to generate from a single prompt. */
  num_return_sequences?: number;
};

export type AssistingGenerationConfig = {
  /**  the lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number update. */
  assistant_confidence_threshold?: number;
  /** the defined candidates number to be generated by draft model/prompt lookup in case of static strategy candidates number update. */
  num_assistant_tokens?: number;
  /** is maximum ngram to use when looking for matches in the prompt. */
  max_ngram_size?: number;
  /** whether to apply chat_template for non-chat scenarios */
  apply_chat_template?: boolean;
};

export type GenericGenerationConfig = {
  // adapters?: AdapterConfig | None
  /** if set to true, the model will echo the prompt in the output. */
  echo?: boolean;
  /** token_id of <eos> (end of sentence) */
  eos_token_id?: number;
  /** reduces absolute log prob as many times as the token was generated. */
  frequency_penalty?: number;
  /** if set to true, then generation will not stop even if <eos> token is met. */
  ignore_eos?: boolean;
  /** if set to true stop string that matched generation will be included in generation output (default?: false) */
  include_stop_str_in_output?: boolean;
  /** number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
        Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default?: 0). */
  logprobs?: number;
  /** Maximum length the generated tokens can have.
   * Corresponds to the length of the input prompt max_new_tokens.
   * Its effect is overridden by `max_new_tokens`, if also set.
   * */
  max_length?: number;
  /** Maximum numbers of tokens to generate, excluding the number of tokens in the prompt.
   * `max_new_tokens` has priority over `max_length`. */
  max_new_tokens?: number;
  /** set 0 probability for eos_token_id for the first eos_token_id generated tokens. */
  min_new_tokens?: number;
  /** reduces absolute log prob if the token was generated at least once. */
  presence_penalty?: number;
  /** the parameter for repetition penalty. 1.0 means no penalty. */
  repetition_penalty?: number;
  /** a set of strings that will cause pipeline to stop generating further tokens. */
  stop_strings?: Set<string>;
  /** a set of tokens that will cause pipeline to stop generating further tokens. */
  stop_token_ids?: Set<number>;
};

export type StructuredOutputGenerationConfig = {
  /** This object is used to store the configuration for structured generation, which includes
   * the JSON schema and other related parameters. */
  structured_output_config?: StructuredOutputConfig;
};

export type DecodedResultsConfig = {
  /** a helper option to get DecodedResult from LLMPipeline and keep backward compability.
   * If set to true, LLMPipeline.generate() will return DecodedResults object instead of string.
   * If set to false, LLMPipeline.generate() will return default value.
   */
  return_decoded_results?: boolean;
};

/** Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
 * and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
 * be used while greedy and beam search parameters will not affect decoding at all.
 */
export type GenerationConfig = GenericGenerationConfig &
  BeamSearchGenerationConfig &
  RandomSamplingsGenerationConfig &
  AssistingGenerationConfig &
  StructuredOutputGenerationConfig &
  DecodedResultsConfig;

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
};
