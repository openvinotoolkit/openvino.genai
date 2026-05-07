from .llava import LLAVAInputsPreprocessor, NanoLlavaInputsPreprocessor
from .minicpmv import MiniCPMVInputsPreprocessor
from .minicpmo import MiniCPMOInputsPreprocessor
from .internvl import InternVLInputsPreprocessor
from .phi3 import Phi3MMInputsPreprocessor
from .phi4 import Phi4MMInputsPreprocessor
from .qwen2 import Qwen2VLInputsPreprocessor
from .qwen3 import Qwen3VLInputsPreprocessor
from .gemma3 import Gemma3InputsPreprocessor

MODEL_TYPE_TO_CLS_MAPPING = {
    "qwen3_vl": Qwen3VLInputsPreprocessor,
    "qwen2_vl_text": Qwen2VLInputsPreprocessor,
    "qwen2_vl": Qwen2VLInputsPreprocessor,
    "qwen2_5_vl": Qwen2VLInputsPreprocessor,
    "qwen2_5_vl_text": Qwen2VLInputsPreprocessor,
    "llava": LLAVAInputsPreprocessor,
    "gemma3": Gemma3InputsPreprocessor,
    "phi4mm": Phi4MMInputsPreprocessor,
    "phi4_multimodal": Phi4MMInputsPreprocessor,
    "phi3_v": Phi3MMInputsPreprocessor,
    "minicpmv": MiniCPMVInputsPreprocessor,
    "minicpmo": MiniCPMOInputsPreprocessor,
    "llava_next": LLAVAInputsPreprocessor,
    "llava-qwen2": NanoLlavaInputsPreprocessor,
    "internvl_chat": InternVLInputsPreprocessor,
}

__all__ = [
    "MODEL_TYPE_TO_CLS_MAPPING",
]
