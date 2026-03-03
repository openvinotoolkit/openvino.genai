
from abc import ABC, abstractmethod
import csv


# Registry for evaluators
EVALUATOR_REGISTRY = {}


def register_evaluator(*names):
    def decorate(cls):
        for name in names:
            assert (
                name not in EVALUATOR_REGISTRY
            ), f"Evaluator named '{name}' conflicts with existing evaluators! Please register with a non-conflicting alias instead."

            EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorate


class Evaluator(ABC):
    @abstractmethod
    def dump_gt(self, csv_name: str):
        pass

    @abstractmethod
    def dump_predictions(self, csv_name: str):
        pass

    @abstractmethod
    def score(self, model_or_data, **kwargs):
        pass

    @abstractmethod
    def worst_examples(self, top_k: int = 5, metric="similarity"):
        pass
 
    @abstractmethod
    def get_generation_fn(self):
        raise NotImplementedError("generation_fn should be returned")


class BaseEvaluator(Evaluator):
    @staticmethod
    def _escape_for_csv(df):
        """Escape newlines and tabs in dataframe string values for clean CSV output."""
        def escape_value(value):
            if not isinstance(value, str):
                return value
            # Normalize line endings and escape special characters
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            return value.replace("\n", "\\n").replace("\t", "\\t")
        
        return df.copy().map(escape_value)

    def dump_gt(self, csv_name: str):
        self._escape_for_csv(self.gt_data).to_csv(csv_name, quoting=csv.QUOTE_ALL, index=False)

    def dump_predictions(self, csv_name: str):
        self._escape_for_csv(self.predictions).to_csv(csv_name, quoting=csv.QUOTE_ALL, index=False)
