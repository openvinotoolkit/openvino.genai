import datasets
from transformers import set_seed
from transformers import AutoProcessor
from typing import Any, Union

import pandas as pd
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING

from .registry import register_evaluator, BaseEvaluator
from .text_evaluator import TextEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity

def get_pil_from_url(url):
    response = requests.get(url, stream=True).raw
    image = Image.open(response)
    return image.convert("RGB")


def preprocess_fn(example):
    return {"prompts": example["instruction"], "images": get_pil_from_url(example["image_url"])}


def prepare_default_data(num_samples=None):
    DATASET_NAME = "ucla-contextual/contextual_test"
    NUM_SAMPLES = 24 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = datasets.load_dataset(DATASET_NAME, split="test", streaming=True).take(NUM_SAMPLES)
    return default_dataset.map(lambda x: preprocess_fn(x),
                                remove_columns=default_dataset.column_names)


INPUT_MAKER_REGISTRY = {}


def register_input_maker(*names):
    def decorate(cls):
        for name in names:
            assert (
                name not in INPUT_MAKER_REGISTRY
            ), f"Input maker '{name}' conflicts with existing evaluators! Please register with a non-conflicting alias instead."

            INPUT_MAKER_REGISTRY[name] = cls
        return cls
    return decorate


@register_input_maker("llava", "llava_next")
def prepare_llava_inputs(processor, image, prompt):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return processor(images=[image], text=prompt, return_tensors="pt")


def autodetect_language(model):
    model2language = {
        "chatglm": "cn",
        "qwen2": "cn",
        "qwen": "cn",
        "baichuan": "cn",
        "minicpmv": "cn",
        "internlm": "cn",
        "llava-qwen2": "cn"
    }

    if not hasattr(model, "config"):
        return "en"
    return model2language.get(model.config.model_type, "en")


@register_evaluator(
    "visual-text"
)
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
        language=None,
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
            language=language,
            gen_answer_fn=gen_answer_fn,
            generation_config=generation_config,
            seqs_per_request=seqs_per_request,
        )


    def dump_gt(self, csv_name: str):
        self.gt_data.to_csv(csv_name)

    def score(self, model, gen_answer_fn=None):
        predictions = self._generate_data(model, gen_answer_fn)

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
        def default_gen_answer(model, prompt, image, processor, tokenizer, max_new_tokens, crop_question):
            #input_maker = INPUT_MAKER_REGISTRY[model.config.model_type]
            #inputs = input_maker(processor, image, prompt)
            preprocess_inputs = MODEL_TYPE_TO_CLS_MAPPING[model.config.model_type].preprocess_inputs
            #print("Inputs: ", prompt, image)

            inputs = preprocess_inputs(prompt, image, processor, tokenizer)
            
            # image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
            # tokens = model.generate(**inputs, images=image_tensor, do_sample=False, max_new_tokens=max_new_tokens)
            tokens = model.generate(**inputs,  do_sample=False, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
            if crop_question:
                tokens = tokens[:, inputs["input_ids"].shape[-1] :]

            answer = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
            #print("Result: ", answer)
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
            if self.language is None:
                print(
                    "No language detecting in the base model or ground truth data. Taking language from target model."
                )
                self.language = autodetect_language(model)

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
        df["language"] = self.language

        return df
