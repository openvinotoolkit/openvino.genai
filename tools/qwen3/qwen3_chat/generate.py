from typing import Any

import torch
import transformers

from .model import SYSTEM_PROMPT


def init_history() -> list[dict[str, Any]]:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]


def _extract_text_ids(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sequences"):
        return output.sequences
    return _extract_text_ids(output[0])


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

    gen_kwargs: dict[str, Any] = {
        "streamer": transformers.TextStreamer(
            processor,
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "thinker_do_sample": False,
    }
    if speaker:
        gen_kwargs["speaker"] = speaker

    input_len = inputs["input_ids"].shape[-1]

    if enable_audio:
        gen_kwargs["return_audio"] = True
        gen_kwargs["talker_do_sample"] = True
        text_ids, audio = model.generate(**inputs, **gen_kwargs)
        text_ids = _extract_text_ids(text_ids)
    else:
        gen_kwargs["return_audio"] = False
        output = model.generate(**inputs, **gen_kwargs)
        text_ids = _extract_text_ids(output)
        audio = None

    generated_ids = text_ids[:, input_len:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return text, audio
