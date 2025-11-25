from typing import Any, Union

import os
import random
import tarfile
import datasets

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Literal
from itertools import zip_longest
from transformers import set_seed
from transformers.image_utils import load_image

from .registry import register_evaluator
from .text_evaluator import TextEvaluator
from .utils import get_ignore_parameters_flag

DEF_VIDEO_FRAMES_AMOUNT = 10


def preprocess_fn(example):
    return {
        "prompts": example["instruction"],
        "images": load_image(example["image_url"]),
        "videos": None,
    }


def prepare_default_data_image(num_samples=None):
    DATASET_NAME = "ucla-contextual/contextual_test"
    NUM_SAMPLES = 24 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: preprocess_fn(x), remove_columns=default_dataset.column_names
    )


def prepare_default_data_video(num_samples=None, num_frames=DEF_VIDEO_FRAMES_AMOUNT):
    from huggingface_hub import hf_hub_download
    from transformers.video_utils import load_video

    DATASET_NAME = "lmms-lab/LLaVA-Video-178K"
    SUBSET = "30_60_s_academic_v0_1"
    NUM_SAMPLES = 24 if num_samples is None else num_samples

    questions_per_video_set = datasets.load_dataset(DATASET_NAME, SUBSET,
                                                    split="open_ended",
                                                    data_files={"open_ended": f"{SUBSET}/30_60_s_academic_oe_v0_1_qa_processed.json"})
    questions_per_video = {val['video']: val for val in questions_per_video_set}

    # 30_60_s_academic_v0_1_videos_10.tar.gz - just the most lightweight chunk among subset
    # https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/30_60_s_academic_v0_1
    # the archive contains 56 videos
    videos_arc_path = hf_hub_download(repo_id="lmms-lab/LLaVA-Video-178K",
                                      filename=f"{SUBSET}/{SUBSET}_videos_10.tar.gz",
                                      repo_type="dataset")

    video_samples = []
    extract_dir = "./videos"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(videos_arc_path, "r:gz") as tar:
        all_videos = tar.getnames()

        random.seed(42)  # nosec
        video_samples = random.sample(all_videos, NUM_SAMPLES)  # nosec
        for sample in video_samples:
            tar.extract(sample, path=extract_dir)

    # if num_frames < total_num_frames, sample each total_num_frames/num_frames frames or sample all frames
    def default_sample_indices_fn(metadata, **kwargs):
        total_num_frames = metadata.total_num_frames
        if num_frames < total_num_frames:
            return np.arange(0, total_num_frames, total_num_frames / num_frames, dtype=int)
        return np.arange(0, total_num_frames, dtype=int)

    data = []
    for video_rel_path in video_samples:
        video_tensor = load_video(os.path.join(extract_dir, video_rel_path), backend="opencv", sample_indices_fn=default_sample_indices_fn)
        prompt = questions_per_video[video_rel_path]['conversations'][0]['value'].replace("<image>\n", "")
        data.append({'prompts': prompt, "images": None, 'videos': video_tensor[0]})

    return data


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


@register_evaluator("visual-text", "visual-video-text")
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
        task_type: Literal['visual-text', 'visual-video-text'] = "visual-text",
        frames_num: int | None = None,
    ) -> None:
        self.processor = processor
        self.is_image_input = (task_type == "visual-text")
        self.frames_num = frames_num or DEF_VIDEO_FRAMES_AMOUNT
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
            model, prompt, image, video, processor, tokenizer, max_new_tokens, crop_question
        ):

            from optimum.intel.openvino.modeling_visual_language import \
                MODEL_TYPE_TO_CLS_MAPPING
            preprocess_inputs = MODEL_TYPE_TO_CLS_MAPPING[
                model.config.model_type
            ].preprocess_inputs
            inputs = preprocess_inputs(prompt, image, processor, tokenizer, config=model.config, video=video)
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
                    assert "videos" in self.test_data
                    data = dict(self.test_data)
                data = pd.DataFrame.from_dict(data)
        else:
            input_data = prepare_default_data_image(self.num_samples) if self.is_image_input else prepare_default_data_video(self.num_samples, self.frames_num)
            data = pd.DataFrame.from_dict(input_data)

        prompt_data = data["prompts"]
        image_data = data["images"]
        videos_data = data["videos"]

        answers = []
        prompts = prompt_data.values
        images = image_data.values
        videos = videos_data.values

        for p, i, v in tqdm(zip_longest(prompts, images, videos), desc="Evaluate pipeline"):
            answers.append(
                gen_answer_fn(
                    model,
                    p,
                    i,
                    v,
                    self.processor,
                    self.tokenizer,
                    self.max_new_tokens,
                    self._crop_question,
                )
            )

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)

        return df
