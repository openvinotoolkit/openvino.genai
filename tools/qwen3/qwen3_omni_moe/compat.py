import json
from pathlib import Path
from typing import Any

from .constants import ARCHITECTURE_PATCHES, MODEL_TYPE_PATCHES


def _needs_config_patch(ckpt: str | Path) -> bool:
    config_path = Path(ckpt) / "config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        raw = json.load(f)
    return raw.get("model_type") in MODEL_TYPE_PATCHES


def _is_dense_model(ckpt: str | Path) -> bool:
    config_path = Path(ckpt) / "config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        raw = json.load(f)
    talker_text = raw.get("talker_config", {}).get("text_config", {})
    return talker_text.get("num_experts", 0) == 0


def _patch_config_json(ckpt: str | Path) -> None:
    config_path = Path(ckpt) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    _patch_model_types(cfg)
    _patch_architectures(cfg)
    _set_dense_experts(cfg)

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def _patch_model_types(cfg: dict[str, Any]) -> None:
    if cfg.get("model_type") in MODEL_TYPE_PATCHES:
        cfg["model_type"] = MODEL_TYPE_PATCHES[cfg["model_type"]]

    for sub_key in ("thinker_config", "talker_config"):
        sub = cfg.get(sub_key, {})
        if sub.get("model_type") in MODEL_TYPE_PATCHES:
            sub["model_type"] = MODEL_TYPE_PATCHES[sub["model_type"]]
        text_cfg = sub.get("text_config", {})
        if text_cfg.get("model_type") in MODEL_TYPE_PATCHES:
            text_cfg["model_type"] = MODEL_TYPE_PATCHES[text_cfg["model_type"]]


def _patch_architectures(cfg: dict[str, Any]) -> None:
    if "architectures" in cfg:
        cfg["architectures"] = [ARCHITECTURE_PATCHES.get(a, a) for a in cfg["architectures"]]


def _set_dense_experts(cfg: dict[str, Any]) -> None:
    for sub_key in ("thinker_config", "talker_config"):
        text_cfg = cfg.get(sub_key, {}).get("text_config", {})
        if "num_experts" not in text_cfg:
            text_cfg["num_experts"] = 0
            text_cfg["num_experts_per_tok"] = 1
            text_cfg["decoder_sparse_step"] = 1


def _patch_talker_classes() -> None:
    from torch import nn
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        ACT2FN,
        Qwen3OmniMoeTalkerConfig,
        Qwen3OmniMoeTalkerDecoderLayer,
        Qwen3OmniMoeTalkerResizeMLP,
        Qwen3OmniMoeTalkerTextMLP,
        Qwen3OmniMoeThinkerTextAttention,
        Qwen3OmniMoeThinkerTextRMSNorm,
    )

    def _patched_layer_init(self: Any, config: Any, layer_idx: int) -> None:
        super(Qwen3OmniMoeTalkerDecoderLayer, self).__init__()
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeTalkerTextMLP(config)
        self.input_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def _patched_resize_init(self: Any, config: Qwen3OmniMoeTalkerConfig) -> None:
        super(Qwen3OmniMoeTalkerResizeMLP, self).__init__()
        proj_dim = config.thinker_hidden_size
        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, proj_dim, bias=True)
        self.linear_fc2 = nn.Linear(proj_dim, config.text_config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.text_config.hidden_act]

    Qwen3OmniMoeTalkerDecoderLayer.__init__ = _patched_layer_init
    Qwen3OmniMoeTalkerResizeMLP.__init__ = _patched_resize_init


def apply_dense_compat(ckpt: str | Path) -> None:
    if _needs_config_patch(ckpt):
        print("Detected dense (non-MoE) qwen3_omni model, patching config...")
        _patch_config_json(ckpt)

    if _is_dense_model(ckpt):
        print("Applying dense talker compatibility patches...")
        _patch_talker_classes()
