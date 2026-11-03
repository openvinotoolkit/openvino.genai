// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Parser } from "./parsers.js";
import type { Tensor } from "openvino-node";
import type { VideoGenerationPerfMetrics } from "./perfMetrics.js";

export enum StreamingStatus {
  RUNNING,
  STOP,
  CANCEL,
  TOOL_CALL_STOP,
}

export enum GenerationFinishReason {
  NONE,
  STOP,
  LENGTH,
  TOOL_CALL,
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

/** Unsigned integer value represented as JS `number` whenever possible; if the value is too large for `number`, `bigint` is used.
 *
 * For size_t-like fields this corresponds to a maximum of `2^32 - 1` on 32-bit systems
 * and `2^64 - 1` on 64-bit systems.
 */
export type Uint = number | bigint;

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
  num_beams?: Uint;
  /** number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_beam_groups?: Uint;
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
  num_return_sequences?: Uint;
  /** if set to int > 0, all ngrams of that size can only occur once.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  no_repeat_ngram_size?: Uint;
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
  top_k?: Uint;
  /** if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. */
  top_p?: number;
  /** initializes random generator.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  rng_seed?: Uint;
};

export type AssistingGenerationConfig = {
  /**  the lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number update. */
  assistant_confidence_threshold?: number;
  /** the defined candidates number to be generated by draft model/prompt lookup in case of static strategy candidates number update.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  num_assistant_tokens?: Uint;
  /** is maximum ngram to use when looking for matches in the prompt.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_ngram_size?: Uint;
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
  eos_token_id?: Uint;
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
  logprobs?: Uint;
  /** Maximum length the generated tokens can have. Corresponds to the length of the input prompt max_new_tokens.
   * Its effect is overridden by `max_new_tokens`, if also set.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_length?: Uint;
  /** Maximum numbers of tokens to generate, excluding the number of tokens in the prompt.
   * `max_new_tokens` has priority over `max_length`.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_new_tokens?: Uint;
  /** set 0 probability for eos_token_id for the first eos_token_id generated tokens.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  min_new_tokens?: Uint;
  /** reduces absolute log prob if the token was generated at least once. */
  presence_penalty?: number;
  /** the parameter for repetition penalty. 1.0 means no penalty. */
  repetition_penalty?: number;
  /** a set of strings that will cause pipeline to stop generating further tokens. */
  stop_strings?: Set<string>;
  /** a set of tokens that will cause pipeline to stop generating further tokens. */
  stop_token_ids?: Set<Uint>;
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

/** Generation config for WhisperPipeline. Extends GenerationConfig with Whisper-specific options. */
export type WhisperGenerationConfig = GenerationConfig & {
  /** Language for generation: a plain code (e.g. "en") or the wrapped token form (e.g. "<|en|>"). For multilingual models only. */
  language?: string;
  /** Task: "translate" or "transcribe". For multilingual models only. */
  task?: string;
  /** If true, return timestamps for text segments. */
  return_timestamps?: boolean;
  /** If true, return word-level timestamps (requires word_timestamps in pipeline constructor). */
  word_timestamps?: boolean;
  /** Initial prompt to steer spelling or style. */
  initial_prompt?: string;
  /** Hotwords to steer spelling or style. */
  hotwords?: string;
  /** Encoder attention alignment heads for word-level timestamps. Each pair is [layer_index, head_index].
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  alignment_heads?: [Uint, Uint][];
  /** Decoder start token id (e.g. startoftranscript).
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  decoder_start_token_id?: Uint;
  /** Padding token id.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  pad_token_id?: Uint;
  /** No timestamps token id.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  no_timestamps_token_id?: Uint;
  /** Transcribe task token id.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  transcribe_token_id?: Uint;
  /** Translate task token id.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  translate_token_id?: Uint;
  /** Start of previous segment token id.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  prev_sot_token_id?: Uint;
  /** Language token to id map (from generation_config.json).
   *
   * @type Map values use `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  lang_to_id?: Record<string, Uint>;
  /** Max initial timestamp index.
   *
   * @type Uses `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems. */
  max_initial_timestamp_index?: Uint;
  /** Whether the model is multilingual. */
  is_multilingual?: boolean;
  /** Token ids to suppress at the beginning.
   *
   * @type Array elements use `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  begin_suppress_tokens?: Uint[];
  /** Token ids to suppress.
   *
   * @type Array elements use `number` whenever possible; if an integer value is too large for `number`, `bigint` is returned.
   * Maximum value is `2^32 - 1` on 32-bit systems and `2^64 - 1` on 64-bit systems.
   */
  suppress_tokens?: Uint[];
};

/**
 * Generation config for ASRPipeline. Extends the Whisper generation config with
 * Qwen3-ASR-specific options. Whisper-specific fields are ignored by Qwen3-ASR models
 * and vice versa.
 */
export type ASRGenerationConfig = WhisperGenerationConfig & {
  /** Context passed as a previous transcription to steer spelling or style. Supported by Qwen3-ASR models only. */
  context?: string;
  /** If true, apply inverse text normalization (ITN). Only honored by SenseVoiceSmall. */
  use_itn?: boolean;
};

/** Generation config for Text2SpeechPipeline. Extends GenerationConfig with speech-specific options. */
export type SpeechGenerationConfig = GenerationConfig & {
  /** Minimum ratio of output length to input text length; prevents output that's too short. */
  minlenratio?: number;
  /** Maximum ratio of output length to input text length; prevents excessively long outputs. */
  maxlenratio?: number;
  /** Probability threshold for stopping decoding; when output probability exceeds this, generation stops. */
  threshold?: number;
};

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

export type WhisperPipelineProperties = {
  schedulerConfig?: SchedulerConfig;
} & Record<string, unknown>;

export type ASRPipelineProperties = {
  schedulerConfig?: SchedulerConfig;
} & Record<string, unknown>;

export type ImageGenerationConfig = {
  /** Additional prompt for the second text encoder. */
  prompt_2?: string;
  /** Additional prompt for the third text encoder. */
  prompt_3?: string;
  /** Negative prompt for the primary text encoder. */
  negative_prompt?: string;
  /** Negative prompt for the second text encoder. */
  negative_prompt_2?: string;
  /** Negative prompt for the third text encoder. */
  negative_prompt_3?: string;
  /** Number of images to generate for a single prompt. */
  num_images_per_prompt?: Uint;
  /** Seed for the default random generator. */
  rng_seed?: Uint;
  /** Guidance scale used during denoising. */
  guidance_scale?: number;
  /** Output image height in pixels. */
  height?: number;
  /** Output image width in pixels. */
  width?: number;
  /** Number of denoising steps. */
  num_inference_steps?: Uint;
  /** Maximum sequence length for T5-based text encoders. */
  max_sequence_length?: number;
  /** Strength parameter for img2img/inpainting-compatible configs. */
  strength?: number;
};

export type Text2ImagePipelineProperties = Record<string, unknown>;

export type Image2ImagePipelineProperties = Record<string, unknown>;

export type InpaintingPipelineProperties = Record<string, unknown>;

/**
 * Video generation config. Fields that are not set keep their current values.
 * Optional fields can be set to `null` to unset them.
 */
export type VideoGenerationConfig = {
  /**
   * Negative prompt. Used only with classifier-free guidance, which is on when `guidance_scale`
   * or the audio guidance scale (after the model defaults are applied) is above 1.
   */
  negative_prompt?: string | null;
  /** Number of videos to generate per `generate()` call. */
  num_videos_per_prompt?: Uint;
  /** Guidance scale. Classifier-free guidance is on when it is above 1. */
  guidance_scale?: number;
  /** Output video height in pixels. -1 uses the model default. */
  height?: number;
  /** Output video width in pixels. -1 uses the model default. */
  width?: number;
  /** Number of denoising steps. -1 uses the model default. */
  num_inference_steps?: number;
  /** Maximum sequence length of the text encoder and tokenizer. -1 uses the model default. */
  max_sequence_length?: number;
  /**
   * Guidance rescale factor that fixes overexposure with zero terminal SNR
   * (https://arxiv.org/pdf/2305.08891.pdf). 0 or lower disables rescaling.
   * Unset uses the model default, which is 0 (disabled) for LTX-Video.
   */
  guidance_rescale?: number | null;
  /** Number of video frames to generate. 0 uses the model default, which is 161 for LTX-Video. */
  num_frames?: Uint;
  /**
   * Video frame rate, which affects the rope interpolation scale.
   * Unset uses the model default, which is 25 for LTX-Video.
   */
  frame_rate?: number | null;
  /**
   * Guidance scale for the audio modality. Unset uses the model's own default: LTX-2.3 has one (7.0)
   * that is independent of `guidance_scale`, so the audio modality can turn on classifier-free guidance
   * on its own. Models without an audio default, such as LTX-2.0, fall back to `guidance_scale`.
   * The other `audio_*` fields work the same way.
   */
  audio_guidance_scale?: number | null;
  /**
   * (LTX-2.3) Spatio-Temporal Guidance scale. Adds a transformer pass that perturbs the blocks in
   * `spatio_temporal_guidance_blocks`. 0 disables it.
   */
  stg_scale?: number | null;
  /** Spatio-Temporal Guidance scale for the audio modality. 0 disables the audio term. */
  audio_stg_scale?: number | null;
  /**
   * (LTX-2.3) Modality isolation guidance scale. Adds a transformer pass without the audio-to-video and
   * video-to-audio cross attention. 1 or lower disables it. The pass is shared with `audio_modality_scale`,
   * but each modality is guided only when its own scale is above 1.
   */
  modality_scale?: number | null;
  /** Modality isolation guidance scale for the audio modality. 1 or lower disables the audio term. */
  audio_modality_scale?: number | null;
  /** Guidance rescale factor for the audio modality. */
  audio_guidance_rescale?: number | null;
  /**
   * Transformer block indices perturbed by Spatio-Temporal Guidance.
   * Indices outside the model's block range are ignored.
   */
  spatio_temporal_guidance_blocks?: number[] | null;
};

export type Text2VideoPipelineProperties = Record<string, unknown>;

export type VideoGenerationResult = {
  /** Generated video tensor with shape `[num_videos, num_frames, H, W, 3]` and `u8` RGB pixels. */
  video: Tensor;
  /** Performance metrics of the call that produced the video. */
  perfMetrics: VideoGenerationPerfMetrics;
  /**
   * Generated audio waveform with shape `[num_videos, num_channels, num_samples]` and `f32` samples.
   * `undefined` for models without audio.
   */
  audio?: Tensor;
  /** Sample rate of `audio` in Hz, `0` for models without audio. */
  audioSampleRate: number;
};

/**
 * Callback for image generation, called once per denoising step.
 *
 * The callback may be synchronous or asynchronous. When it returns a Promise,
 * generation waits for it to settle before the next step, so an in-callback
 * `await pipeline.decode(latent)` completes without blocking the event loop.
 *
 * @param step       Current step index (0-based).
 * @param numSteps   Total number of denoising steps.
 * @param latent     Current latent image. Pass it to `pipeline.decode(latent)` to obtain the intermediate image.
 * @returns `true` to stop generation early, `false` to continue (optionally wrapped in a Promise).
 */
export type ImageGenerationCallback = (
  step: number,
  numSteps: number,
  latent: Tensor,
) => boolean | Promise<boolean>;
/**
 * Callback for text-to-image generation, called once per denoising step.
 *
 * @deprecated Use `ImageGenerationCallback` instead, as text-to-image pipelines now support the same callback signature as other image generation pipelines.
 */
export type Text2ImageCallback = ImageGenerationCallback;

export type Text2SpeechPipelineProperties = Record<string, unknown>;

/**
 * Device and scheduler properties for OmniPipeline initialization.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type OmniPipelineProperties = {
  schedulerConfig?: SchedulerConfig;
} & Record<string, unknown>;

/**
 * Speech-side generation config for the Qwen3-Omni talker.
 *
 * Controls the speech output stack (Talker + CodePredictor + Code2Wav). Standalone config:
 * the thinker text decode is steered by a separate {@link GenerationConfig} passed as `textConfig`.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type OmniTalkerSpeechConfig = {
  /** Enable speech output. Defaults to true; set false to produce text only. */
  return_audio?: boolean;
  /**
   * Speaker identity: either a name looked up in the model config, or an explicit
   * embedding tensor (`[1, 1, talker_hidden_size]`, f32). Default selects the model's default speaker.
   */
  speaker?: string | Tensor;
  /**
   * Number of codec frames accumulated before streaming each audio chunk. Must be >= 1.
   * For the standard Qwen3-Omni each frame is 80ms of audio at 24 kHz (1920 samples), and the
   * first chunk is 555 samples shorter because the vocoder trims its warmup once per utterance.
   */
  audio_chunk_frames?: Uint;
  /**
   * Number of earlier codec frames the vocoder decodes together with each streamed chunk, so the
   * chunk starts with the history a non-streamed decode would see. 0 decodes every chunk on its
   * own. Larger values give smoother seams but cost more vocoder work. Must be <= 4096. Default 25.
   */
  audio_chunk_left_context_frames?: Uint;
  /** Cap on talker autoregressive steps. Independent of `textConfig.max_new_tokens`. */
  max_new_tokens?: Uint;
  /** RNG seed for talker + CodePredictor sampling. */
  rng_seed?: Uint;
  /** Talker sampling temperature override (must be > 0 when set). */
  talker_temperature?: number;
  /** Talker top-k override (must be >= 1 when set). */
  talker_top_k?: Uint;
  /** Talker repetition penalty override (must be > 0 when set; 1.0 = no penalty). */
  talker_repetition_penalty?: number;
  /** CodePredictor sampling temperature override (must be > 0 when set). */
  cp_temperature?: number;
  /** CodePredictor top-k override (must be >= 1 when set). */
  cp_top_k?: Uint;
};

/**
 * Metadata describing the original video source. Controls video frame sampling before encoding.
 *
 * @note This is a preview API and is subject to change in future releases.
 */
export type VideoMetadata = {
  /** Frame rate of the original video in frames per second. 0 (default) means unknown. */
  fps?: number;
  /**
   * Indices of frames to sample from the provided video tensor.
   * When omitted or empty, model-specific sampling is applied if defined, otherwise all frames are processed.
   * When non-empty, only the specified frames are extracted and model-specific sampling logic is skipped.
   */
  frames_indices?: Uint[];
};
