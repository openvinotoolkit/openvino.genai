from pathlib import Path
from typing import Any
import gc
import types

import torch
import torch.nn.functional as F
import openvino as ov
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from transformers import Qwen3OmniMoeForConditionalGeneration

from .constants import AUDIO_ENCODER_NAME
from .utils import cleanup_torchscript_cache


@torch.no_grad()
def _compute_aftercnn_time(audio: Any, num_mels: int, time_in: int) -> int:
    dummy = torch.randn([1, 1, num_mels, time_in], dtype=torch.float32)
    dummy = audio.conv2d1(dummy)
    dummy = audio.conv2d2(dummy)
    dummy = audio.conv2d3(dummy)
    return dummy.shape[3]


def _forward_wrap_audio_encoder(
    self: Any,
    padded_feature: torch.Tensor,
    padded_mask_after_cnn: torch.Tensor,
    aftercnn_lens: torch.Tensor,
) -> torch.Tensor:
    padded_feature = padded_feature.unsqueeze(1)
    padded_embed = F.gelu(self.conv2d1(padded_feature))
    padded_embed = F.gelu(self.conv2d2(padded_embed))
    padded_embed = F.gelu(self.conv2d3(padded_embed))
    b, c, f, t = padded_embed.size()
    padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    positional_embedding = (
        self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    hidden_states = padded_embed[padded_mask_after_cnn]

    window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
    cu_chunk_lens: list[int] = [0]
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

    for encoder_layer in self.layers:
        layer_outputs = encoder_layer(hidden_states, cu_seqlens)
        hidden_states = layer_outputs[0]

    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states


def convert_audio_encoder(model: Qwen3OmniMoeForConditionalGeneration, output_dir: Path) -> None:
    output_path = output_dir / AUDIO_ENCODER_NAME
    if output_path.exists():
        return

    print("Convert audio encoder model...")
    audio = model.thinker.audio_tower
    audio.forward = types.MethodType(_forward_wrap_audio_encoder, audio)
    __make_16bit_traceable(audio)

    num_mels = audio.num_mel_bins
    time_in = 200
    aftercnn_time = _compute_aftercnn_time(audio, num_mels, time_in)

    ov_model = ov.convert_model(
        audio,
        example_input={
            "padded_feature": torch.randn([3, num_mels, time_in], dtype=torch.float32),
            "padded_mask_after_cnn": torch.ones([3, aftercnn_time], dtype=torch.bool),
            "aftercnn_lens": torch.tensor([aftercnn_time] * 3, dtype=torch.long),
        },
    )
    ov.save_model(ov_model, output_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print("Audio encoder model successfully converted")
