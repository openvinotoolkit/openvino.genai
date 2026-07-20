import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Qwen3VLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 151655)
        else:
            self.def_image_token_id = 151655

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional[Union["Image", list["Image"]]] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional[Union["VideoInput", list["VideoInput"]]] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]
        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "transformers" in str(type(model)):
            return inputs

        model.rope_deltas = None

        embeds, attn, positional_ids, visual_positional_ids_masks, deepstack = model.get_multimodal_embeddings(
            **inputs,
            cache_position=torch.arange(full_tokenized_chat.shape[1]),
        )

        num_visual_before = int(visual_positional_ids_masks[:, :prefix_len].sum().item())
        num_visual_after = int(visual_positional_ids_masks[:, prefix_len:].sum().item())

        cropped_visual_pos_masks = None
        cropped_deepstack = None
        if num_visual_after > 0:
            cropped_visual_pos_masks = visual_positional_ids_masks[:, prefix_len:]
            if deepstack is not None:
                cropped_deepstack = []
                for layer_idx, layer_embed in enumerate(deepstack):
                    le = layer_embed if torch.is_tensor(layer_embed) else torch.as_tensor(np.asarray(layer_embed))
                    # crop to [num_visual_after, hidden]
                    le_seg = le[num_visual_before:]
                    cropped_deepstack.append(le_seg.tolist())

        model._prefill_visual_pos_masks = cropped_visual_pos_masks
        model._prefill_position_ids = positional_ids
        model._prefill_deepstack_visual_embeds = cropped_deepstack
        model._prefill_rope_deltas = model.rope_deltas
        new_inputs = {
            "inputs_embeds": embeds,
            "attention_mask": attn if attn is not None else inputs["attention_mask"],
        }

        return new_inputs


class Qwen3_5VLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 248056)
        else:
            self.def_image_token_id = 248056

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional[Union["Image", list["Image"]]] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional[Union["VideoInput", list["VideoInput"]]] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]
        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs
