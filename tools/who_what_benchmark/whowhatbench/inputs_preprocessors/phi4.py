import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union
import inspect

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Phi4MMInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": answer})

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
        if video is not None:
            raise ValueError("Video input is not supported")

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        audio_token = getattr(processor.tokenizer, "audio_token", "<|audio_1|>")

        if audio is not None and audio_token not in text:
            text = audio_token + text

        self.update_images(image)
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, _ in enumerate(image):
                image_token = getattr(processor.tokenizer, "image_token", f"<|image_{i + 1}|>")
                if image_token not in text:
                    text = image_token + text

        text_prompt = ""
        if processor.tokenizer.chat_template is None:
            if self.chat_mode:
                text_hist = ""
                for msg in self.chat_history:
                    if msg["role"] == "user":
                        text_hist += user_prompt + msg["content"] + prompt_suffix
                    elif msg["role"] == "assistant":
                        text_hist += assistant_prompt + msg["content"] + prompt_suffix
                text_prompt = text_hist + assistant_prompt
            else:
                if text.startswith(user_prompt):
                    text_prompt = text
                else:
                    text_prompt = user_prompt + text + prompt_suffix + assistant_prompt
        else:
            if self.chat_mode:
                self.chat_history.append({"role": "user", "content": text})
                conversation = self.chat_history
            else:
                conversation = [{"role": "user", "content": text}]

            text_prompt = processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

        # TODO: audio to chat_mode
        audio_input = {}
        if "audio" in inspect.signature(processor.__call__).parameters:
            sample_rate = None
            if isinstance(audio, tuple):
                audio, sample_rate = audio
            if isinstance(audio, list) and len(audio) == 1 and isinstance(audio[0], tuple):
                audio, sample_rate = audio[0]
            audio_input["audio"] = audio
            if sample_rate is not None:
                audio_input["sampling_rate"] = sample_rate
        else:
            audio_input["audios"] = audio

        inputs = processor(text=text_prompt, images=self.images, **audio_input, return_tensors="pt")
        return inputs
