# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file includes utility functions copied from the MileBench repository:
# https://github.com/MileBench/MileBench
#
# Licensed under the Apache License

import os
import json
import re
import openvino

import numpy as np
from PIL import Image
from rouge import Rouge


class MileBenchDataset:
    def __init__(self, data_dir, subset, subset_size=200):
        self.data_dir = data_dir
        self.subset = subset
        self.subset_size = subset_size

        annotation_path = os.path.join(
            self.data_dir, self.subset, f"{self.subset}.json"
        )
        with open(annotation_path, "r") as f:
            self.annotation = json.load(f)

        self.image_dir = os.path.join(self.data_dir, self.subset, "images")

    def __len__(self):
        return min(self.annotation["meta_data"]["num_sample"], self.subset_size)

    @staticmethod
    def _transform_string(s: str) -> str:
        import re

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
        qwen2_vl_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        for i in range(img_num):
            rmv_txt = "{image#%d}" % (i + 1)
            rmv_tbl = "{table#%d}" % (i + 1)
            context = context.replace(rmv_txt, qwen2_vl_image_placeholder)
            context = context.replace(rmv_tbl, qwen2_vl_image_placeholder)

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
                re.search(self.commaStrip, inText) != None
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
