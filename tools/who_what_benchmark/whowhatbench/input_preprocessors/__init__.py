from .preprocessors import fix_phi3_v_eos_token_id
from .llava_preprocessors import LLAVAInputsPreprocessor, NanoLlavaInputsPreprocessor
from .minicpmv_preprocessors import MiniCPMVInputsPreprocessor
from .minicpmo_preprocessors import MiniCPMOInputsPreprocessor
from .internvl_preprocessors import InternVLInputsPreprocessor
from .phi3_preprocessors import Phi3MMInputsPreprocessor
from .phi4_preprocessors import Phi4MMInputsPreprocessor
from .qwen2_preprocessors import Qwen2VLInputsPreprocessor
from .qwen3_preprocessors import Qwen3VLInputsPreprocessor
from .gemma3_preprocessors import Gemma3InputsPreprocessor

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
    "fix_phi3_v_eos_token_id",
]
