import argparse
import difflib
import os
import json
import pandas as pd
from PIL import Image
import logging
from datasets import load_dataset
from diffusers import DiffusionPipeline
from optimum.intel.openvino import OVModelForCausalLM
from optimum.utils import NormalizedConfigManager, NormalizedTextConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from optimum.exporters.tasks import TasksManager
from optimum.intel import OVPipelineForText2Image

import openvino_genai
from whowhatbench import EVALUATOR_REGISTRY, MODELTYPE2TASK


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = (
    TasksManager._SUPPORTED_MODEL_TYPE["llama"]
)
NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers",
    num_attention_heads="num_attention_heads",
)


class GenAIModelWrapper:
    """
    A helper class to store additional attributes for GenAI models
    """

    def __init__(self, model, model_dir, model_type):
        self.model = model
        self.model_type = model_type

        if model_type == "text":
            self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        elif model_type == "text-to-image":
            self.config = DiffusionPipeline.load_config(model_dir, trust_remote_code=True)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.model, attr)


def load_text_genai_pipeline(model_dir, device="CPU"):
    try:
        import openvino_genai
    except ImportError:
        logger.error("Failed to import openvino_genai package. Please install it.")
        exit(-1)
    logger.info("Using OpenVINO GenAI API")
    return GenAIModelWrapper(openvino_genai.LLMPipeline(model_dir, device), model_dir, "text")


def load_text_model(
    model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if ov_config:
        with open(ov_config) as f:
            ov_options = json.load(f)
    else:
        ov_options = None

    if use_hf:
        logger.info("Using HF Transformers API")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map=device.lower()
        )
    elif use_genai:
        model = load_text_genai_pipeline(model_id, device)
    else:
        try:
            model = OVModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device=device, ov_config=ov_options
            )
        except ValueError:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = OVModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_options,
            )

    return model


TEXT2IMAGE_TASK2CLASS = {
    "text-to-image": OVPipelineForText2Image,
}


def load_text2image_genai_pipeline(model_dir, device="CPU"):
    try:
        import openvino_genai
    except ImportError:
        logger.error("Failed to import openvino_genai package. Please install it.")
        exit(-1)
    logger.info("Using OpenVINO GenAI API")
    return GenAIModelWrapper(
        openvino_genai.Text2ImagePipeline(model_dir, device),
        model_dir,
        "text-to-image"
    )


def load_text2image_model(
    model_type, model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    if ov_config:
        with open(ov_config) as f:
            ov_options = json.load(f)
    else:
        ov_options = None

    if use_genai:
        model = load_text2image_genai_pipeline(model_id, device)
    elif use_hf:
        model = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True)
    else:
        TEXT2IMAGEPipeline = TEXT2IMAGE_TASK2CLASS[model_type]

        try:
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id, trust_remote_code=True, device=device, ov_config=ov_options
            )
        except ValueError:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = TEXT2IMAGEPipeline.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                use_cache=True,
                device=device,
                ov_config=ov_options,
            )

    return model


def load_model(
    model_type, model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False
):
    from .registry import MODELTYPE2TASK

    if model_id is None:
        return None

    if model_type == "text":
        return load_text_model(model_id, device, ov_config, use_hf, use_genai)
    elif MODELTYPE2TASK[model_type] == "text-to-image":
        return load_text2image_model(
            model_type, model_id, device, ov_config, use_hf, use_genai
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_prompts(args):
    if args.dataset is None:
        return None
    split = "validation"
    if args.split is not None:
        split = args.split
    if "," in args.dataset:
        path_name = args.dataset.split(",")
        path = path_name[0]
        name = path_name[1]
    else:
        path = args.dataset
        name = None
    data = load_dataset(path=path, name=name, split=split)

    res = data[args.dataset_field]

    res = {"prompts": list(res)}

    return res


def parse_args():
    parser = argparse.ArgumentParser(
        prog="WWB CLI",
        description="This sript generates answers for questions from csv file",
    )

    parser.add_argument(
        "--base-model",
        default=None,
        help="Model to ground truth generation.",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Model to comparison with base_model. Usually it is compressed, quantized version of base_model.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer for divergency metric. If not defined then will be load from base_model or target_model.",
    )

    parser.add_argument(
        "--gt-data",
        default=None,
        help="CSV file with base_model generation. If defined and exists then base_model will not used."
        "I defined and not exists them will be generated by base_model evaluation.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["text", "text-to-image"],
        default="text",
        help="Indicated the model type, e.g. 'text' - for causal text generation, 'text-to-image' - for image generation.",
    )
    parser.add_argument(
        "--data-encoder",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Model for measurement of similarity between base_model and target_model."
        "By default it is sentence-transformers/all-mpnet-base-v2,"
        "but for Chinese LLMs better to use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of the dataset with prompts. The interface for dataset is load_dataset from datasets library."
        "Please provide this argument in format path,name (for example wikitext,wikitext-2-v1)."
        "If None then internal list of prompts will be used.",
    )
    parser.add_argument(
        "--dataset-field",
        type=str,
        default="text",
        help="The name of field in dataset for prompts. For example question or context in squad."
        "Will be used only if dataset is defined.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split of prompts from dataset (for example train, validation, train[:32])."
        "Will be used only if dataset is defined.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory name for saving the per sample comparison and metrics in CSV files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of prompts to use from dataset",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print results and their difference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device to run the model, e.g. 'CPU', 'GPU'.",
    )
    parser.add_argument(
        "--ov-config",
        type=str,
        default=None,
        help="Path to the JSON file that contains OpenVINO Runtime configuration.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "cn"],
        default=None,
        help="Used to select default prompts based on the primary model language, e.g. 'en', 'ch'.",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use AutoModelForCausalLM from transformers library to instantiate the model.",
    )
    parser.add_argument(
        "--genai",
        action="store_true",
        help="Use LLMPipeline from transformers library to instantiate the model.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Text-to-image specific parameter that defines the image resolution.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Text-to-image specific parameter that defines the number of denoising steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Text-to-image specific parameter that defines the seed value.",
    )

    return parser.parse_args()


def check_args(args):
    assert not (args.base_model is None and args.target_model is None)
    assert not (args.base_model is None and args.gt_data is None)


def load_tokenizer(args):
    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
    elif args.base_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, trust_remote_code=True
        )

    return tokenizer


def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    if use_loguru_colors:
        green = "<GREEN><black>"
        red = "<RED><black>"
        endgreen = "</black></GREEN>"
        endred = "</black></RED>"
    else:
        green = "\x1b[38;5;16;48;5;2m"
        red = "\x1b[38;5;16;48;5;1m"
        endgreen = "\x1b[0m"
        endred = "\x1b[0m"

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(a[a0:a1])
        elif opcode == "insert":
            output.append(f"{green}{b[b0:b1]}{endgreen}")
        elif opcode == "delete":
            output.append(f"{red}{a[a0:a1]}{endred}")
        elif opcode == "replace":
            output.append(f"{green}{b[b0:b1]}{endgreen}")
            output.append(f"{red}{a[a0:a1]}{endred}")
    return "".join(output)


def genai_gen_answer(model, tokenizer, question, max_new_tokens, skip_question):
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    out = model.generate(question, config)
    return out


def genai_gen_image(model, prompt, num_inference_steps, generator=None):
    image_tensor = model.generate(
        prompt,
        width=model.resolution[0],
        height=model.resolution[1],
        num_inference_steps=num_inference_steps,
        random_generator=generator
    )
    image = Image.fromarray(image_tensor.data[0])
    return image


def get_evaluator(base_model, args):
    # config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # task = TasksManager.infer_task_from_model(config._name_or_path)
    # TODO: Add logic to auto detect task based on model_id (TaskManager does not work for locally saved models)
    task = MODELTYPE2TASK[args.model_type]

    try:
        EvaluatorCLS = EVALUATOR_REGISTRY[task]
        prompts = load_prompts(args)

        if task == "text-generation":
            tokenizer = load_tokenizer(args)
            return EvaluatorCLS(
                base_model=base_model,
                gt_data=args.gt_data,
                test_data=prompts,
                tokenizer=tokenizer,
                similarity_model_id=args.data_encoder,
                num_samples=args.num_samples,
                language=args.language,
                gen_answer_fn=genai_gen_answer if args.genai else None,
            )
        elif task == "text-to-image":
            return EvaluatorCLS(
                base_model=base_model,
                gt_data=args.gt_data,
                test_data=prompts,
                num_samples=args.num_samples,
                resolution=(args.image_size, args.image_size),
                num_inference_steps=args.num_inference_steps,
                gen_image_fn=genai_gen_image if args.genai else None,
                is_genai=args.genai,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    except KeyError:
        raise ValueError(
            f"Attempted to load evaluator for '{task}', but no evaluator for this model type found!"
            "Supported model types: {', '.join(EVALUATOR_REGISTRY.keys())}"
        )


def print_text_results(evaluator):
    metric_of_interest = "similarity"
    worst_examples = evaluator.worst_examples(top_k=5, metric=metric_of_interest)
    for i, e in enumerate(worst_examples):
        ref_text = ""
        actual_text = ""
        diff = ""
        for l1, l2 in zip(
            e["source_model"].splitlines(), e["optimized_model"].splitlines()
        ):
            if l1 == "" and l2 == "":
                continue
            ref_text += l1 + "\n"
            actual_text += l2 + "\n"
            diff += diff_strings(l1, l2) + "\n"

        logger.info(
            "--------------------------------------------------------------------------------------"
        )
        logger.info("## Reference text %d:\n%s", i + 1, ref_text)
        logger.info("## Actual text %d:\n%s", i + 1, actual_text)
        logger.info("## Diff %d: ", i + 1)
        logger.info(diff)


def print_image_results(evaluator):
    metric_of_interest = "similarity"
    worst_examples = evaluator.worst_examples(top_k=1, metric=metric_of_interest)
    for i, e in enumerate(worst_examples):
        logger.info(
            "--------------------------------------------------------------------------------------"
        )
        logger.info(f"Top-{i+1} example:")
        logger.info(e)


def main():
    args = parse_args()
    check_args(args)

    if args.gt_data and os.path.exists(args.gt_data):
        evaluator = get_evaluator(None, args)
    else:
        base_model = load_model(
            args.model_type,
            args.base_model,
            args.device,
            args.ov_config,
            args.hf,
            args.genai,
        )
        evaluator = get_evaluator(base_model, args)

        if args.gt_data:
            evaluator.dump_gt(args.gt_data)
        del base_model

    if args.target_model:
        target_model = load_model(
            args.model_type,
            args.target_model,
            args.device,
            args.ov_config,
            args.hf,
            args.genai,
        )
        all_metrics_per_question, all_metrics = evaluator.score(
            target_model, evaluator.get_generation_fn() if args.genai else None
        )
        logger.info("Metrics for model: %s", args.target_model)
        logger.info(all_metrics)

        if args.output:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            df = pd.DataFrame(all_metrics_per_question)
            df.to_csv(os.path.join(args.output, "metrics_per_qustion.csv"))
            df = pd.DataFrame(all_metrics)
            df.to_csv(os.path.join(args.output, "metrics.csv"))

    if args.verbose and args.target_model is not None:
        if args.model_type == "text":
            print_text_results(evaluator)
        elif "text-to-image" in args.model_type:
            print_image_results(evaluator)


if __name__ == "__main__":
    main()
