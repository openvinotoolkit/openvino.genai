#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import json
import os
import re
import requests
import shutil
from logging import getLogger
from typing import Optional

import numpy as np
from PIL import Image
from rouge import Rouge
from tqdm import tqdm

import openvino
from openvino_genai import (
    AggregationMode,
    CacheEvictionConfig,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
)

logger = getLogger(__name__)


class MileBenchDataset:
    def __init__(self, data_dir, subset, subset_size=200):
        self.data_dir = data_dir
        self.subset = subset
        self.subset_size = subset_size

        self._download_data()
        annotation_path = os.path.join(self.data_dir, self.subset, f"{self.subset}.json")
        with open(annotation_path) as f:
            self.annotation = json.load(f)

        self.image_dir = os.path.join(self.data_dir, self.subset, "images")

    def _download_data(self):
        LINKS = {
            "MileBench_part0.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part0.tar.gz",
            "MileBench_part1.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part1.tar.gz",
            "MileBench_part2.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part2.tar.gz",
            "MileBench_part3.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part3.tar.gz",
            "MileBench_part4.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part4.tar.gz",
            "MileBench_part5.tar.gz": "https://huggingface.co/datasets/FreedomIntelligence/MileBench/resolve/main/MileBench_part5.tar.gz",
        }

        SUBSET2ARCHIVE = {
            # Realistic Temporal
            "ActionLocalization": "MileBench_part0.tar.gz",
            "ActionPrediction": "MileBench_part0.tar.gz",
            "ActionSequence": "MileBench_part0.tar.gz",
            "CharacterOrder": "MileBench_part0.tar.gz",
            "CounterfactualInference": "MileBench_part1.tar.gz",
            "EgocentricNavigation": "MileBench_part1.tar.gz",
            "MovingAttribute": "MileBench_part2.tar.gz",
            "MovingDirection": "MileBench_part2.tar.gz",
            "ObjectExistence": "MileBench_part3.tar.gz",
            "ObjectInteraction": "MileBench_part3.tar.gz",
            "ObjectShuffle": "MileBench_part3.tar.gz",
            "SceneTransition": "MileBench_part3.tar.gz",
            "StateChange": "MileBench_part3.tar.gz",
            # Realistic Semantic
            "ALFRED": "MileBench_part0.tar.gz",
            "CLEVR-Change": "MileBench_part1.tar.gz",
            "DocVQA": "MileBench_part1.tar.gz",
            "IEdit": "MileBench_part2.tar.gz",
            "MMCoQA": "MileBench_part2.tar.gz",
            "MultiModalQA": "MileBench_part2.tar.gz",
            "nuscenes": "MileBench_part3.tar.gz",
            "OCR-VQA": "MileBench_part4.tar.gz",
            "SlideVQA": "MileBench_part4.tar.gz",
            "Spot-the-Diff": "MileBench_part4.tar.gz",
            "TQA": "MileBench_part5.tar.gz",
            "WebQA": "MileBench_part5.tar.gz",
            "WikiVQA": "MileBench_part5.tar.gz",
            # Diagnostic
            "TextNeedleInAHaystack": "MileBench_part5.tar.gz",
            "ImageNeedleInAHaystack": "MileBench_part2.tar.gz",
            "GPR1200": "MileBench_part1.tar.gz",
        }

        archive_name = SUBSET2ARCHIVE.get(self.subset)
        archive_url = LINKS[archive_name]
        archive_path = os.path.join(self.data_dir, archive_name)
        dir_name = os.path.join(self.data_dir, self.subset)

        if not os.path.exists(dir_name):
            if not os.path.exists(archive_path):
                logger.info(f"Downloading {archive_name} from {archive_url}...")
                os.makedirs(self.data_dir, exist_ok=True)
                response = requests.get(archive_url, stream=True)
                response.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded archive to {archive_path}")
            else:
                logger.info(f"Archive already exists at {archive_path}")

            logger.info(f"Extracting {archive_path}...")
            shutil.unpack_archive(archive_path, self.data_dir)
            logger.info(f"Extracted to {self.data_dir}")
        else:
            logger.info(f"Already extracted to {self.data_dir}")

    def __len__(self):
        return min(self.annotation["meta_data"]["num_sample"], self.subset_size)

    @staticmethod
    def _transform_string(s: str) -> str:
        counter = iter(range(1, s.count("{i}") + 1))
        return re.sub(r"\{i\}", lambda _: str(next(counter)), s)

    @staticmethod
    def _preprocess_image(image_path, max_size=512, min_size=32):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if max(w, h) > max_size:
            scale_factor = max_size / max(w, h)
        elif min(w, h) < min_size:
            scale_factor = min_size / min(w, h)
        else:
            scale_factor = 1.0  # No scaling needed

        new_size = (int(w * scale_factor), int(h * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of range for the dataset.")

        ann = self.annotation["data"][idx]
        task_instructions = self.annotation["meta_data"]["task_instruction"]

        context = ann["task_instance"]["context"]
        if "choice_list" in ann["task_instance"].keys():
            choice_str = "\nChoice list: \n"
            choice_str += "\n".join(
                [
                    (f"{chr(65+idx)}. ") + f"{item}"
                    for idx, item in enumerate(ann["task_instance"]["choice_list"])
                ]
            )
            choice_str += "\nYour answer is: "
            context += choice_str

        img_num = len(ann["task_instance"]["images_path"])

        def idx_to_ov_image_placeholder(idx: int) -> str:
            return f"<ov_genai_image_{idx}>"

        for i in range(img_num):
            rmv_txt = "{image#%d}" % (i + 1)
            rmv_tbl = "{table#%d}" % (i + 1)
            image_placeholder = idx_to_ov_image_placeholder(i)
            context = context.replace(rmv_txt, image_placeholder)
            context = context.replace(rmv_tbl, image_placeholder)

        task_instruction_id = ann["task_instruction_id"]
        context_str = task_instructions[task_instruction_id] + "\n" + context
        prompt = MileBenchDataset._transform_string(context_str)

        images = []
        for p in ann["task_instance"]["images_path"]:
            img_path = os.path.join(self.image_dir, p)
            image = MileBenchDataset._preprocess_image(img_path)
            image_data = np.array(image)
            image_tensor = openvino.Tensor(image_data)
            images.append(image_tensor)

        return {
            "prompt": prompt,
            "images": images,
            "gt_answer": ann["response"],
            "choice_list": ann["task_instance"].get("choice_list", None),
        }


class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index + 65)
        elif index < 52:
            return "A" + chr(index + 65 - 26)
        else:
            return "B" + chr(index + 65 - 26 - 26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('"')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self, predictions):
        rouge = Rouge()
        acc = []
        for res in predictions:
            gt_ans = self.process(res["gt_answer"])
            pred_ans = self.process(res["pred"])
            assert gt_ans != ""
            if pred_ans == "":
                score = 0
            else:
                score = rouge.get_scores(pred_ans, gt_ans)[0]["rouge-l"]["f"]
            acc.append(score)
        return np.mean(acc)

    def match_choice(self, text, option):
        """Return: A B C D..."""

        def preprocess_option_string(option_string):
            # First, preprocess the option text to normalize it
            processed_option = self.process(option_string)

            # Then, escape any special regex characters in the processed option text
            # List of regex special characters that need to be escaped
            special_chars = [
                "\\",
                ".",
                "^",
                "$",
                "*",
                "+",
                "?",
                "{",
                "}",
                "[",
                "]",
                "|",
                "(",
                ")",
            ]
            # Escape the special characters by prefixing them with a backslash
            for char in special_chars:
                if char in processed_option:
                    processed_option = processed_option.replace(char, "\\" + char)
            # escaped_option = escape_special_chars(processed_option)
            return processed_option

        if text == "":
            return "C"
        try:
            # Maybe start from the head
            # 1. Char+Choice: `A. Blastomycosis`
            option_str = "|".join(
                [preprocess_option_string(f"{k} {v}") for k, v in option.items()]
            )
            option_pattern = rf"({option_str})"
            option_res = re.search(
                option_pattern, text, re.S
            )  # NOTE we dont use match_all
            if option_res:
                return (option_res.group(0)[0]).upper()

            # 2. Choice: `Blastomycosis`
            option_str = "|".join(
                [
                    preprocess_option_string(v).replace(" ", "")
                    for k, v in option.items()
                ]
            )
            option_pattern = rf"({option_str})"
            option_res = re.search(
                option_pattern, text.replace(" ", ""), re.S
            )  # NOTE we dont use match_all
            if option_res:
                for k, v in option.items():
                    if option_res[0].strip() == preprocess_option_string(v).replace(
                        " ", ""
                    ):
                        return k.upper()

            # 3. Char: `A` `AB`
            if len(text) in [1, 2] and text.upper() in option.keys():
                return text.upper()

            # use gpt extract

        except Exception as e:
            print(f"something wrong during match_choice {text}: {e}")
            return text
        return "".join([i.upper() for i in text if i.upper() in option])

    def judge_multi_choice(self, sample):
        gt_ans = sample["gt_answer"]
        pred_ans = sample["pred"]
        choice_list = sample["choice_list"]
        assert gt_ans in choice_list
        # Convert choice_list to a dictionary format expected by match_choice
        option_dict = {self.char(i): choice for i, choice in enumerate(choice_list)}

        # Use match_choice to determine the selected answer from pred_ans
        selected_answer = self.match_choice(pred_ans, option_dict)

        # Check if the selected answer matches the ground truth
        gt_ans_chr = self.char(choice_list.index(sample["gt_answer"]))
        if selected_answer == gt_ans_chr:
            return 1, selected_answer
        else:
            return 0, selected_answer

    def process_sample(self, sample):
        sample["gt_answer"] = self.process(sample["gt_answer"])
        sample["pred"] = self.process(sample["pred"])
        for i in range(len(sample["choice_list"])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, predictions):
        correct = 0
        for sample in predictions:
            self.process_sample(sample)
            score, extracted_answer = self.judge_multi_choice(sample)
            sample["extracted"] = extracted_answer
            sample["result"] = score
            correct += score
        return correct / len(predictions)

    def evaluate_needle(self, predictions, needle=True):
        correct = 0
        for sample in predictions:
            gt_ans = self.process(sample["gt_answer"])
            pred_ans = self.process(sample["pred"])

            if needle:
                score = 1 if gt_ans in pred_ans.split() else 0
            else:
                score = 1 if gt_ans in pred_ans else 0

            sample["result"] = score
            correct += score
        return correct / len(predictions)

    def evaluate(self, predictions, dataset_name, question_type):
        if "NeedleInAHaystack" in dataset_name or "MMCoQA" in dataset_name:
            return self.evaluate_needle(
                predictions, needle="NeedleInAHaystack" in dataset_name
            )
        elif question_type == "open-ended":
            return self.evaluate_rouge(predictions)
        elif question_type == "multi-choice":
            return self.evaluate_multichoice(predictions)
        else:
            raise ValueError("Dataset not supported")


def get_scheduler_config(num_kv_blocks: Optional[int]) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if num_kv_blocks is not None:
        scheduler_config.num_kv_blocks = num_kv_blocks
        scheduler_config.max_num_batched_tokens = 32 * num_kv_blocks
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.max_num_seqs = 256
    scheduler_config.use_cache_eviction = False
    return scheduler_config


def main():
    parser = ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model_dir", type=str, help="Path to the model directory")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=512, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    parser.add_argument("-s", "--subset", type=str, help="MileBench subset to use")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to MileBench data directory. If not provided, data will be downloaded to ./milebench_data"
    )
    parser.add_argument("--enable_cache_eviction", action='store_true', help="Whether to apply cache eviction")
    parser.add_argument(
        "--num_kv_blocks",
        type=int,
        default=500,
        help=(
            "Number of blocks to statically pre-allocate in the KV cache. "
            "If unspecified, blocks are allocated dynamically based on generation length."
        )
    )
    parser.add_argument("--seqs_per_request", type=int, default=1, help="Number of sequences per request")

    args = parser.parse_args()

    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = False
    generation_config.apply_chat_template = True

    scheduler_config = get_scheduler_config(args.num_kv_blocks)
    if args.enable_cache_eviction:
        scheduler_config.use_cache_eviction = True
        eviction_config = CacheEvictionConfig(
            start_size=32,
            recent_size=64,
            max_cache_size=512,
            aggregation_mode=AggregationMode.SUM,
            snapkv_window_size=8,
        )
        scheduler_config.cache_eviction_config = eviction_config
        print("Eviction is ON")
    else:
        print("Eviction is OFF")

    model_cb = ContinuousBatchingPipeline(args.model_dir, scheduler_config, args.device)

    data = MileBenchDataset(
        data_dir=args.data_dir if args.data_dir is not None else "milebench_data",
        subset=args.subset,
        subset_size=100,
    )

    with tqdm(total=len(data)) as progress_bar:
        prompts, images = [], []
        answers = []
        ref_answers = []
        for p_idx, data_sample in enumerate(data):
            prompt = data_sample["prompt"]
            image = data_sample["images"]

            progress_bar.update(1)
            prompts.append(prompt)
            images.append(image)
            answers.append({"gt_answer": data_sample["gt_answer"], "choice_list": data_sample["choice_list"]})
            ref_answers.append({"gt_answer": data_sample["gt_answer"], "choice_list": data_sample["choice_list"]})

            if len(prompts) == args.seqs_per_request or p_idx == len(data) - 1:
                ans_batch = model_cb.generate(
                    prompts, images=images, generation_config=[generation_config] * len(prompts)
                )

                batch_start_idx = p_idx - len(prompts) + 1
                for i, output in enumerate(ans_batch, start=batch_start_idx):
                    answers[i]["pred"] = output.texts[0]
                prompts.clear()
                images.clear()

    question_type = data.annotation['meta_data']['question_type']
    scorer = Eval()
    score = scorer.evaluate(answers, args.subset, question_type)
    print(f"Score: {score}")

    pipeline_metrics = model_cb.get_metrics()
    print(f"Cache usage: max {pipeline_metrics.max_cache_usage:.3f}, avg {pipeline_metrics.avg_cache_usage:.3f}")


if __name__ == '__main__':
    main()
