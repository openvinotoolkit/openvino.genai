# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file includes utility functions copied from the LongBench repository:
# https://github.com/THUDM/LongBench
#
# Copyright (c) 2023 THU-KEG & Zhipu AI
# Licensed under the MIT License
import re
import string

from collections import Counter
from rouge import Rouge


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    from fuzzywuzzy import fuzz
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def evaluate(model_output, task):
    predictions, answers = [], []
    for data in model_output:
        predictions.append(data["pred"])
        answers.append(data["answers"])
        all_classes = data["all_classes"]
    score = scorer(task, predictions, answers, all_classes)
    return score


def build_chat(prompt, model_name):
    if "Llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "Llama" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "Phi-3" in model_name:
        prompt = f"<|user|>\n{prompt} <|end|>\n<|assistant|>"
    return prompt


def preprocess_prompt(data_sample, subset, model_name):
    prompt_format = dataset2prompt[subset]
    prompt = prompt_format.format(**data_sample)
    if subset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat(prompt, model_name)
    return prompt


def post_process_pred(pred, subset, model_name):
    if subset in ["samsum", "qsum", "hotpotqa", "qasper"] and "Llama-3" in model_name:
        pred = pred[:pred.find("assistant")]
    elif subset == "samsum":
        pred = pred[:pred.find("\nDialogue")]
    elif "Phi-3" in model_name and subset == "hotpotqa":
        pred = pred.lstrip('\n').split('\n')[0]
    elif subset in ["trec", "hotpotqa", "qasper"] and "Qwen" in model_name:
        pred = pred[:pred.find("\nQuestion")]
    return pred
