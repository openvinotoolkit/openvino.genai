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
