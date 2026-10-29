from .llava import LLAVAInputsPreprocessor, NanoLlavaInputsPreprocessor
from .lfm3_vl import Lfm3VLInputsPreprocessor
from .minicpmv import MiniCPMV4_7InputsPreprocessor, MiniCPMVInputsPreprocessor
from .minicpmo import MiniCPMOInputsPreprocessor
from .minicpmv4_6 import MiniCPMV4_6InputsPreprocessor
from .internvl import InternVLInputsPreprocessor
from .phi3 import Phi3MMInputsPreprocessor
from .phi4 import Phi4MMInputsPreprocessor
from .qwen2 import Qwen2VLInputsPreprocessor
from .qwen3 import Qwen3VLInputsPreprocessor, Qwen3_5VLInputsPreprocessor, Qwen3OmniInputsPreprocessor
from .qwen4_exp import Qwen4_ExpInputsPreprocessor
from .gemma3 import Gemma3InputsPreprocessor
from .gemma4 import Gemma4InputsPreprocessor, Gemma4UnifiedInputsPreprocessor, Gemma3nInputsPreprocessor
from .muse_glimmer import MuseGlimmerInputsPreprocessor
from .deepseek_ocr2 import DeepseekOCR2InputsPreprocessor
from .vlm_inputs_preprocessor import VLMInputsPreprocessor

MODEL_TYPE_TO_CLS_MAPPING = {
    "lfm3_vl": Lfm3VLInputsPreprocessor,
    "muse_glimmer": MuseGlimmerInputsPreprocessor,
    "qwen3_vl": Qwen3VLInputsPreprocessor,
    "qwen3_5_moe": Qwen3_5VLInputsPreprocessor,
    "qwen3_5": Qwen3_5VLInputsPreprocessor,
    "qwen3_omni_moe": Qwen3OmniInputsPreprocessor,
    "qwen3_omni": Qwen3OmniInputsPreprocessor,
    "qwen2_vl_text": Qwen2VLInputsPreprocessor,
    "qwen2_vl": Qwen2VLInputsPreprocessor,
    "qwen2_5_vl": Qwen2VLInputsPreprocessor,
    "qwen2_5_vl_text": Qwen2VLInputsPreprocessor,
    "qwen4_exp": Qwen4_ExpInputsPreprocessor,
    "llava": LLAVAInputsPreprocessor,
    "gemma3": Gemma3InputsPreprocessor,
    "gemma4_unified": Gemma4UnifiedInputsPreprocessor,
    "gemma4": Gemma4InputsPreprocessor,
    "gemma3n": Gemma3nInputsPreprocessor,
    "phi4mm": Phi4MMInputsPreprocessor,
    "phi4_multimodal": Phi4MMInputsPreprocessor,
    "phi3_v": Phi3MMInputsPreprocessor,
    "minicpmv": MiniCPMVInputsPreprocessor,
    "minicpmv4_6": MiniCPMV4_6InputsPreprocessor,
    "minicpmv4_7": MiniCPMV4_7InputsPreprocessor,
    "minicpmo": MiniCPMOInputsPreprocessor,
    "llava_next": LLAVAInputsPreprocessor,
    "llava-qwen2": NanoLlavaInputsPreprocessor,
    "internvl_chat": InternVLInputsPreprocessor,
    "deepseek_ocr2": DeepseekOCR2InputsPreprocessor,
}

__all__ = ["MODEL_TYPE_TO_CLS_MAPPING", "VLMInputsPreprocessor"]
