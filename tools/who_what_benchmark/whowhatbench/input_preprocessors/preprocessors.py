import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    __version__,
)
from abc import ABC, abstractmethod
from packaging.version import Version
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


TRANSFORMERS_VERSION = Version(__version__)


def fix_phi3_v_eos_token_id(model_type: str, tokenizer: PreTrainedTokenizer) -> dict:
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
        self.videos = None
        self.chat_history = []
        self.chat_mode = chat_mode

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
