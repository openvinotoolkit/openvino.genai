"""
Metrics for text similarity
"""

from difflib import SequenceMatcher
from transformers import AutoTokenizer
from PIL import Image
import torch
import torch.nn.functional as F

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm


def evaluate_similarity(model, data_gold, data_prediction):
    answers_gold = data_gold["answers"].values
    answers_prediction = data_prediction["answers"].values

    metric_per_question = []
    for gold, prediction in tqdm(
        zip(answers_gold, answers_prediction), desc="Similarity evaluation"
    ):
        embeddings = model.encode([gold, prediction])
        cos_sim = util.cos_sim(embeddings, embeddings)
        metric_per_question.append(cos_sim[0, 1].item())

    metric_dict = {"similarity": np.mean(metric_per_question)}
    return metric_dict, {"similarity": metric_per_question}


def evaluate_divergency(tokenizer, data_gold, data_prediction):
    answers_gold = data_gold["answers"].values
    answers_prediction = data_prediction["answers"].values

    DEBUG = False
    # NOTE: a - reference answers, b - answers to evaluate
    fdt_list = []  # each value = the position of first divergent (different) token.
    sdt_list = []  # each value = number of tokens to correct in the prediction.
    sdtn_list = []  # each value = share of tokens to correct in the prediction
    fdt_max = []  # each value = total number of tokens in the reference
    for a_answer, b_answer in zip(answers_gold, answers_prediction):
        a_indexes = tokenizer.encode(a_answer, return_tensors="pt").squeeze().tolist()
        b_indexes = tokenizer.encode(b_answer, return_tensors="pt").squeeze().tolist()
        if not a_indexes and not b_indexes:
            sdt_list.append(0)
            fdt_list.append(0)
            sdtn_list.append(0)
            fdt_max.append(0)
        elif a_indexes and not b_indexes:
            sdt_list.append(len(a_indexes))
            fdt_list.append(0)
            sdtn_list.append(1)
            fdt_max.append(len(a_indexes))
        elif not a_indexes and b_indexes:
            sdt_list.append(len(b_indexes))
            fdt_list.append(0)
            sdtn_list.append(1)
            fdt_max.append(0)
        else:
            if isinstance(a_indexes, int):
                a_indexes = list([a_indexes])
            if isinstance(b_indexes, int):
                b_indexes = list([b_indexes])
            fdt_max.append(len(a_indexes))

            matcher = SequenceMatcher(None, a_indexes, b_indexes)
            blocks = matcher.get_matching_blocks()
            a, b, size = blocks[0]
            fdt = 0
            if a == 0 and b == 0:
                fdt = blocks[0].size
            fdt_list.append(fdt)

            num_matched = sum(block.size for block in blocks)
            sdt = len(b_indexes) - num_matched
            sdt_list.append(sdt)
            sdt_norm = sdt / len(b_indexes)
            sdtn_list.append(sdt_norm)

            if DEBUG:
                print(blocks)
                for block in blocks:
                    a, b, size = block
                    matched = a_indexes[a : a + size + 1]
                    print(matched)
                    print(tokenizer.decode(matched))
                    matched = b_indexes[b : b + size + 1]
                    print(matched)
                    print(tokenizer.decode(matched))
    fdt_max = np.average(fdt_max)
    metric_per_question = {
        "FDT": fdt_list,
        "SDT": sdt_list,
        "FDT norm": np.array(fdt_list) / fdt_max,
        "SDT norm": sdtn_list,
    }

    fdt_avg = np.average(fdt_list)
    metric_dict = {
        "FDT": fdt_avg,
        "SDT": np.average(sdt_list),
        "FDT norm": fdt_avg / fdt_max,
        "SDT norm": np.average(sdtn_list),
    }

    return metric_dict, metric_per_question


class TextSimilarity:
    def __init__(self, model_id) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
            pad_token = tokenizer.pad_token
        else:
            pad_token = tokenizer.eos_token
        self.model = SentenceTransformer(model_id, tokenizer_kwargs={"pad_token": pad_token}, trust_remote_code=True)

    def evaluate(self, gt, prediction):
        return evaluate_similarity(self.model, gt, prediction)


class TextDivergency:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def evaluate(self, gt, prediction):
        return evaluate_divergency(self.tokenizer, gt, prediction)


# Image metrics
def evaluate_image_similarity(processor, model, data_gold, data_prediction):
    images_gold = data_gold["images"].values
    images_prediction = data_prediction["images"].values

    metric_per_image = []
    for gold, prediction in tqdm(
        zip(images_gold, images_prediction), desc="Image Similarity evaluation"
    ):
        gold_image = Image.open(gold)
        prediction_image = Image.open(prediction)

        gold_inputs = processor(images=gold_image, return_tensors="pt")["pixel_values"]
        prediction_inputs = processor(images=prediction_image, return_tensors="pt")[
            "pixel_values"
        ]

        with torch.no_grad():
            gold_outputs = model.get_image_features(gold_inputs)
            prediction_outputs = model.get_image_features(prediction_inputs)

        cos_sim = F.cosine_similarity(gold_outputs, prediction_outputs)
        print("cos_sim: ", cos_sim.item())
        metric_per_image.append(cos_sim.item())

    metric_dict = {"similarity": np.mean(metric_per_image)}
    return metric_dict, {"similarity": metric_per_image}


class ImageSimilarity:
    def __init__(self, model_id) -> None:
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).eval()

    def evaluate(self, gt, prediction):
        return evaluate_image_similarity(self.processor, self.model, gt, prediction)
