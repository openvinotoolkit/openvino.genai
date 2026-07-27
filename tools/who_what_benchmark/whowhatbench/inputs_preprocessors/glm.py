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


class GlmEdgeVInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the GLM-Edge-V (``model_type == "glm"``) VLM family.

    GLM-Edge-V exposes an ``AutoModelForCausalLM`` architecture
    (``GlmForCausalLM``) with an integrated vision tower. Unlike the models
    already covered by ``optimum.intel`` / whowhatbench mappings, its
    ``AutoProcessor`` resolves to a bare text tokenizer, so the image branch
    has to be driven through a standalone image processor. The chat template
    inserts ``<|begin_of_image|>`` placeholder tokens for every image, and the
    model consumes a 6D ``pixel_values`` tensor produced by an
    ``MllamaImageProcessor``.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode, model)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

    @staticmethod
    def _resolve_image_processor(processor, config):
        # load_processor attaches a dedicated image processor for VLMs whose
        # AutoProcessor collapses to a plain tokenizer. Fall back to loading one
        # from the model config so the preprocessor also works standalone.
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            return image_processor
        if callable(processor) and hasattr(processor, "size"):
            # processor already is an image processor
            return processor
        name_or_path = getattr(config, "_name_or_path", None) if config is not None else None
        if name_or_path:
            return AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=True)
        raise ValueError(
            "GLM-Edge-V requires an image processor, but none could be resolved "
            "from the provided processor or model config."
        )

    @staticmethod
    def _resolve_tokenizer(processor, tokenizer):
        if tokenizer is not None:
            return tokenizer
        inner = getattr(processor, "tokenizer", None)
        if inner is not None:
            return inner
        # processor itself is a tokenizer
        return processor

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
        if processor is None and tokenizer is None:
            raise ValueError("Processor or tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        tok = self._resolve_tokenizer(processor, tokenizer)

        self.update_images(image)

        content = []
        if image is not None:
            imgs = image if isinstance(image, list) else [image]
            content.extend([{"type": "image"}] * len(imgs))
        content.append({"type": "text", "text": text})

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            conversation = self.chat_history
        else:
            conversation = [{"role": "user", "content": content}]

        # The GLM-Edge-V chat template inserts the <|begin_of_image|> placeholder
        # tokens (config.boi_token_id) that the vision bridge replaces with image
        # features during the forward pass.
        text_prompt = tok.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        text_inputs = tok(text_prompt, return_tensors="pt")
        inputs = dict(text_inputs)

        if self.images is not None:
            image_processor = self._resolve_image_processor(processor, config)
            image_inputs = image_processor(images=self.images, return_tensors="pt")
            # GLM-Edge-V's forward consumes only the 6D ``pixel_values`` tensor.
            # The underlying MllamaImageProcessor additionally emits Mllama-only
            # tiling metadata (aspect_ratio_ids/mask, num_tiles) that
            # GlmForCausalLM.generate() rejects as unused model_kwargs, so keep
            # just pixel_values.
            if "pixel_values" in image_inputs:
                inputs["pixel_values"] = image_inputs["pixel_values"]

        return inputs

    def is_image_token(self, tokenized_input: list, idx: int) -> bool:
        boi_token_id = self.def_image_token_id
        if boi_token_id is None:
            return True
        return tokenized_input[idx] == boi_token_id
