import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class YoutuVLInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for tencent Youtu-VL models (model_type="youtu_vl").

    Youtu-VL reuses the Qwen2-VL style multimodal chat interface: media
    placeholders are expressed via <|vision_start|><|image_pad|><|vision_end|>
    in the chat template, and the processor consumes the rendered text together
    with the raw images/videos. The preprocessing is therefore identical in
    structure to Qwen2VLInputsPreprocessor and is kept generic (no Hub-id
    special casing) so it applies to the whole youtu_vl architecture.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", None)

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

        self.update_images(image)
        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]

        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

            if self.chat_mode:
                if self.videos is None:
                    self.videos = []
                self.videos.extend(video)
            else:
                self.videos = video
        elif not self.chat_mode:
            self.videos = None

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=self.images,
            text=text_prompt,
            videos=self.videos,
            return_tensors="pt",
        )

        return inputs
