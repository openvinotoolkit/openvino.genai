from typing import Any

THINKER_EMBEDDING_NAME = "openvino_text_embeddings_model.xml"
THINKER_LANGUAGE_NAME = "openvino_language_model.xml"
AUDIO_ENCODER_NAME = "openvino_audio_encoder_model.xml"
VISION_PATCHER_NAME = "openvino_vision_embeddings_model.xml"
VISION_MERGER_NAME = "openvino_vision_embeddings_merger_model.xml"
TALKER_LANGUAGE_NAME = "openvino_talker_model.xml"
TALKER_EMBEDDING_NAME = "openvino_talker_embeddings_model.xml"
CODE_PREDICTOR_NAME = "openvino_code_predictor_model.xml"
CODE2WAV_NAME = "openvino_code2wav_model.xml"

WEIGHT_FORMAT_TO_NNCF: dict[str, dict[str, Any] | None] = {
    "fp16": None,
    "fp32": None,
    "int8": {"mode": "int8_sym"},
    "int4": {
        "mode": "int4_sym",
        "group_size": 128,
        "ratio": 0.8,
    },
}

ATTN_IMPLEMENTATION = "sdpa"

PKV_INPUT_PREFIX = "past_key_values"
PKV_OUTPUT_PREFIX = "present"
BEAM_IDX_NAME = "beam_idx"
INPUTS_EMBEDS = "inputs_embeds"
LOGITS = "logits"
HIDDEN_STATES = "hidden_states"
CACHE_POSITION = "cache_position"
ATTENTION_MASK = "attention_mask"
POSITION_IDS = "position_ids"
