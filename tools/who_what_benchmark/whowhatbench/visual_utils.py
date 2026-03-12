import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    __version__,
)
from abc import ABC, abstractmethod
from packaging.version import Version
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


TRANSFORMERS_VERSION = Version(__version__)


def fix_phi3_v_eos_token_id(model_type, tokenizer):
    """
    phi3_v configs aren't consistent. Override the default
    eos_token_id with the one from a tokenizer similar to
    an example in
    https://huggingface.co/microsoft/Phi-3.5-vision-instruct
    """
    if "phi3_v" == model_type:
        return {"eos_token_id": tokenizer.eos_token_id}
    else:
        return dict()


class VLMInputsPreprocessor(ABC):
    def __init__(self, chat_mode: bool = False):
        self.images = None
        self.videos = []
        self.chat_history = []
        self.chat_mode = chat_mode

    @abstractmethod
    def preprocess_inputs(
        self,
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        return None

    @abstractmethod
    def update_chat_history_with_answer(self, answer):
        pass


class Qwen3VLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": answer})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image", "image": image})
        if video is not None:
            conversation[0]["content"].insert(0, {"type": "video", "video": video})

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs


class LLAVAInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": answer})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if self.chat_mode and getattr(processor, "chat_template", None) is None:
            raise ValueError("Chat template is not set, but pipeline was run in chat mode.")
        if image is not None and not isinstance(image, list):
            image = [image]

        if getattr(processor, "chat_template", None) is not None:
            templated_prompt = {"role": "user", "content": [{"type": "text", "text": text}]}
            if image is not None:
                for im in image:
                    templated_prompt["content"].append({"type": "image"})
                if self.chat_mode:
                    if self.images is None:
                        self.images = []
                    self.images.extend(image)
                else:
                    self.images = [*image] 

            if self.chat_mode:
                self.chat_history.append(templated_prompt)
                templated_input = self.chat_history
            else:
                templated_input = [templated_prompt]
            prompt = processor.apply_chat_template(templated_input, add_generation_prompt=True, tokenize=False)
        else:
            if image is not None and "<image>" not in text:
                prompt = ("<image>\n") * len(image) + text
                self.images = [*image]
            else:
                prompt = text

        if TRANSFORMERS_VERSION > Version("4.47.99") and getattr(processor, "patch_size", None) is None:
            if (
                getattr(config, "vision_config", None) is not None
                and getattr(config.vision_config, "patch_size", None) is not None
            ):
                processor.patch_size = config.vision_config.patch_size
            else:
                raise ValueError(
                    "Processor does not have `patch_size` attribute. Please fix the processor or provide `patch_size` in the config."
                )

        inputs = processor(images=self.images, text=prompt, return_tensors="pt")
        return inputs


MODEL_TYPE_TO_CLS_MAPPING = {
    "qwen3_vl": Qwen3VLInputsPreprocessor,
    "llava": LLAVAInputsPreprocessor
}
