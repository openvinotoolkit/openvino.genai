from .qwen3 import Qwen3_5VLInputsPreprocessor
from typing import Optional, Any


class Qwen4_ExpInputsPreprocessor(Qwen3_5VLInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 248056)
        else:
            self.def_image_token_id = 248056
