"""
Metrics for text similarity
"""
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer, util
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
    fdt_list, sdt_list, sdtn_list, fdt_max = [], [], [], []

    for a_answer, b_answer in zip(answers_gold, answers_prediction):
        a_indexes = tokenizer.encode(a_answer, return_tensors="pt").squeeze().tolist()
        b_indexes = tokenizer.encode(b_answer, return_tensors="pt").squeeze().tolist()
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
        sdt = (
            len(b_indexes) - num_matched
        )  # how many tokens to correct in the prediction
        sdt_list.append(sdt)
        sdt_norm = sdt / len(b_indexes)  # share of tokens to correct in the prediction
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


class SimilarityMetric:
    def __init__(self, model_id) -> None:
        self.model = SentenceTransformer(model_id)

    def evaluate(self, gt, prediction):
        return evaluate_similarity(self.model, gt, prediction)


class DivergencyMetric:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def evaluate(self, gt, prediction):
        return evaluate_divergency(self.tokenizer, gt, prediction)
