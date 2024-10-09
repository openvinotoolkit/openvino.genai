from abc import ABC, abstractmethod

from optimum.intel import (
    OVLatentConsistencyModelPipeline,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
)


# Registry for evaluators
EVALUATOR_REGISTRY = {}
MODELTYPE2TASK = {
    "text": "text-generation",
    "sd": "image-generation",
    "sd-xl": "image-generation",
    "sd-lcm": "image-generation",
}

TEXT2IMAGE_TASK2CLASS = {
    "sd": OVStableDiffusionPipeline,
    "sd-xl": OVStableDiffusionXLPipeline,
    "sd-lcm": OVLatentConsistencyModelPipeline,
}


def register_evaluator(*names):
    def decorate(cls):
        for name in names:
            assert (
                name not in EVALUATOR_REGISTRY
            ), f"Evaluator named '{name}' conflicts with existing evaluators! Please register with a non-conflicting alias instead."

            EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorate


class BaseEvaluator(ABC):
    @abstractmethod
    def dump_gt(self, csv_name: str):
        pass

    @abstractmethod
    def score(self, model, **kwargs):
        pass

    @abstractmethod
    def worst_examples(self, top_k: int = 5, metric="similarity"):
        pass
