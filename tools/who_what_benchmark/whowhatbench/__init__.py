from .registry import register_evaluator, EVALUATOR_REGISTRY
from .text_evaluator import TextEvaluator
from .text_evaluator import TextEvaluator as Evaluator
from .text2image_evaluator import Text2ImageEvaluator
from .visualtext_evaluator import VisualTextEvaluator
from .imagetext2image import ImageText2ImageEvaluator


__all__ = [
    "Evaluator",
    "register_evaluator",
    "TextEvaluator",
    "Text2ImageEvaluator",
    "VisualTextEvaluator",
    "ImageText2ImageEvaluator",
    "EVALUATOR_REGISTRY",
]
