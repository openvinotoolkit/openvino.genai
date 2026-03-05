from typing import Any

import torch

from .model import SYSTEM_PROMPT


def init_history() -> list[dict[str, Any]]:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]


def generate_response(
    model: Any,
    processor: Any,
    history: list[dict[str, Any]],
    enable_audio: bool,
    speaker: str | None,
) -> tuple[str, torch.Tensor | None]:
    inputs = processor.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    gen_kwargs: dict[str, Any] = {"thinker_do_sample": False}
    if speaker:
        gen_kwargs["speaker"] = speaker

    if enable_audio:
        gen_kwargs["return_audio"] = True
        gen_kwargs["talker_do_sample"] = True
        text_ids, audio = model.generate(**inputs, **gen_kwargs)
    else:
        gen_kwargs["return_audio"] = False
        text_ids = model.generate(**inputs, **gen_kwargs)
        audio = None

    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return text, audio
