import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union, Any
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class VLMInputsPreprocessor(ABC):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        self.images = None
        self.videos = None
        self.chat_history = []
        self.chat_mode = chat_mode
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", None)
        else:
            self.def_image_token_id = None

    @abstractmethod
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
        return None

    @abstractmethod
    def update_chat_history_with_answer(self, answer):
        pass

    def update_images(self, image):
        if self.chat_mode:
            if image is not None:
                if not self.images:
                    self.images = []
                if isinstance(image, list):
                    self.images.extend(image)
                else:
                    self.images.append(image)
        else:
            self.images = image

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        return inputs

    def is_image_token(self, tokenized_input: list, idx: int) -> bool:
        # can't define if it's image token or not, so will treat as image
        if self.def_image_token_id is None:
            return True

        return tokenized_input[idx] == self.def_image_token_id
