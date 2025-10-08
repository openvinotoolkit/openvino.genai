from typing import Any, Union

import os
import datasets
import pandas as pd
from transformers.image_utils import load_image
from tqdm import tqdm
from transformers import set_seed

from .registry import register_evaluator
from .text_evaluator import TextEvaluator
from .utils import get_ignore_parameters_flag


def preprocess_fn(example):
    return {
        "prompts": example["instruction"],
        "images": load_image(example["image_url"]),
    }


def prepare_default_data(num_samples=None):
    DATASET_NAME = "ucla-contextual/contextual_test"
    NUM_SAMPLES = 24 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: preprocess_fn(x), remove_columns=default_dataset.column_names
    )


def fix_phi3_v_eos_token_id(model_type, tokenizer):
    """
    phi3_v configs aren't consistent. Override the default
    eos_token_id with the one from a tokenizer similar to
    an example in
    https://huggingface.co/microsoft/Phi-3.5-vision-instruct
    """
    if 'phi3_v' == model_type:
        return {"eos_token_id": tokenizer.eos_token_id}
    else:
        return dict()


@register_evaluator("visual-text")
class VisualTextEvaluator(TextEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
        gen_answer_fn=None,
        generation_config=None,
        seqs_per_request=None,
    ) -> None:
        self.processor = processor
        super().__init__(
            base_model=base_model,
            tokenizer=tokenizer,
            gt_data=gt_data,
            test_data=test_data,
            metrics="similarity",
            similarity_model_id=similarity_model_id,
            max_new_tokens=max_new_tokens,
            crop_question=crop_question,
            num_samples=num_samples,
            gen_answer_fn=gen_answer_fn,
            generation_config=generation_config,
            seqs_per_request=seqs_per_request,
        )

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, self.generation_config)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_answer_fn=None, generation_config=None):
        def default_gen_answer(
            model, prompt, image, processor, tokenizer, max_new_tokens, crop_question
        ):

            from optimum.intel.openvino.modeling_visual_language import \
                MODEL_TYPE_TO_CLS_MAPPING
            preprocess_inputs = MODEL_TYPE_TO_CLS_MAPPING[
                model.config.model_type
            ].preprocess_inputs
            inputs = preprocess_inputs(prompt, image, processor, tokenizer, config=model.config)
            tokens = model.generate(
                **inputs,
                **fix_phi3_v_eos_token_id(model.config.model_type, tokenizer),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                **get_ignore_parameters_flag()
            )
            if isinstance(tokens, tuple) and isinstance(tokens[0], list) and isinstance(tokens[0][0], str):
                # Some models return a decoded output, like miniCPM-o
                # The output tuple has format (<list of decoded outputs without question/prompt>, <GenerateDecoderOnlyOutput>)
                return tokens[0][0]
            if crop_question:
                tokens = tokens[:, inputs["input_ids"].shape[-1] :]

            answer = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            return answer

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    assert "images" in self.test_data
                    data = dict(self.test_data)
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(prepare_default_data(self.num_samples))

        prompt_data = data["prompts"]
        image_data = data["images"]

        answers = []
        prompts = prompt_data.values
        images = image_data.values

        for p, i in tqdm(zip(prompts, images), desc="Evaluate pipeline"):
            answers.append(
                gen_answer_fn(
                    model,
                    p,
                    i,
                    self.processor,
                    self.tokenizer,
                    self.max_new_tokens,
                    self._crop_question,
                )
            )

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)

        return df
