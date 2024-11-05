from .registry import register_evaluator, MODELTYPE2TASK, EVALUATOR_REGISTRY
from .text_evaluator import TextEvaluator
from .text_evaluator import TextEvaluator as Evaluator
from .text2image_evaluator import Text2ImageEvaluator

__all__ = [
    "Evaluator",
    "register_evaluator",
    "TextEvaluator",
    "Text2ImageEvaluator",
    "MODELTYPE2TASK",
    "EVALUATOR_REGISTRY",
]
