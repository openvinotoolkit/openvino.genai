import numpy as np
from PIL import Image

from .vlm_inputs_preprocessor import VLMInputsPreprocessor


class MuseGlimmerInputsPreprocessor(VLMInputsPreprocessor):
    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def preprocess_inputs(self, text, image=None, processor=None, tokenizer=None, config=None, video=None, audio=None):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        content = []
        if image is not None:
            content.append({"type": "image"})
        if video is not None:
            content.append({"type": "video"})
            video = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in video]
        content.append({"type": "text", "text": text})

        conversation = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return processor(
            text=prompt,
            images=[image] if image is not None else None,
            videos=[video] if video is not None else None,
            return_tensors="pt",
        )
