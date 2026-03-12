from pathlib import Path
import re
import wave

import numpy as np
import torch

AUDIO_SAMPLE_RATE = 24000

MEDIA_PATTERN = re.compile(
    r'/(image|audio|video)\s+(?:"([^"]+)"|(\S+))'
)


def parse_user_input(raw_input: str) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    remaining = raw_input

    for match in MEDIA_PATTERN.finditer(remaining):
        media_type = match.group(1)
        path = match.group(2) or match.group(3)
        content.append({"type": media_type, media_type: path})

    remaining = MEDIA_PATTERN.sub("", remaining).strip()
    if remaining:
        content.append({"type": "text", "text": remaining})

    return content


def save_audio(audio_tensor: torch.Tensor, output_dir: str | Path, turn_number: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"turn_{turn_number:03d}.wav"
    audio_np = audio_tensor.reshape(-1).detach().cpu().float().numpy()

    pcm_data = np.clip(audio_np, -1.0, 1.0)
    pcm_int16 = (pcm_data * 32767).astype(np.int16)

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())

    return filename
