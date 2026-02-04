"""
Metrics for text similarity
"""

from difflib import SequenceMatcher
from transformers import AutoTokenizer
from PIL import Image
import torch
import torch.nn.functional as F

import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPImageProcessor, CLIPModel
from tqdm import tqdm
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity


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
        trust_remote_code = False
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        except Exception:
            trust_remote_code = True
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
            pad_token = tokenizer.pad_token
        else:
            pad_token = tokenizer.eos_token
        self.model = SentenceTransformer(model_id, tokenizer_kwargs={"pad_token": pad_token}, trust_remote_code=trust_remote_code)

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


class EmbedsSimilarity:
    def evaluate(self, data_gold, data_prediction):
        embeds_gold = data_gold["embeds_path"].values
        embeds_prediction = data_prediction["embeds_path"].values

        metric_per_gen = []
        metric_per_passages = []
        for gold, prediction in tqdm(
            zip(embeds_gold, embeds_prediction), desc="Embeds Similarity evaluation"
        ):
            with open(gold, 'rb') as f:
                gold_data = np.load(f)

            with open(prediction, 'rb') as f:
                prediction_data = np.load(f)

            cos_sim_all = cosine_similarity(gold_data, prediction_data)
            cos_sim = np.diag(cos_sim_all)
            metric_per_passages.append(cos_sim)
            metric_per_gen.append(np.mean(cos_sim))

        metric_dict = {"similarity": np.mean(metric_per_gen)}
        return metric_dict, {"similarity": metric_per_gen, "similarity_per_passages": metric_per_passages}


class RerankingSimilarity:
    MISSING_DOCUMENT_PENALTY = 1

    def evaluate(self, data_gold, data_prediction):
        gold_results = data_gold["top_n_scores_path"].values
        prediction_results = data_prediction["top_n_scores_path"].values

        metric_per_query = []
        similarity_per_query = []
        for gold, prediction in tqdm(
            zip(gold_results, prediction_results), desc="Reranking Similarity evaluation"
        ):
            with open(gold, 'rb') as f:
                gold_data = np.load(f)

            with open(prediction, 'rb') as f:
                prediction_data = np.load(f)

            prediction_scores = {int(pred_info[0]): pred_info[1] for pred_info in prediction_data}
            per_query_text = []
            for document_idx, gold_score in gold_data:
                # if documents is not presented in ranking list, let's set 1 as max possible score difference
                scores_diff = self.MISSING_DOCUMENT_PENALTY
                if document_idx in prediction_scores:
                    scores_diff = abs(gold_score - prediction_scores[document_idx])
                per_query_text.append(scores_diff.item())

            metric_per_query.append(per_query_text)
            dist = np.linalg.norm(per_query_text)
            similarity_per_query.append(1 / (1 + dist))

        metric_dict = {"similarity": np.mean(similarity_per_query)}
        return metric_dict, {"similarity": similarity_per_query, "per_text_scores_diff": metric_per_query}


class VideoSimilarity:
    def __init__(self) -> None:
        from transformers import VivitImageProcessor, VivitModel

        self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2")
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2").eval()

        import lpips

        # alex - faster; vgg - more rigorous assessments; to check when collecting statistics
        self.lpips_model = lpips.LPIPS(net="alex").to("cpu")

    def load_video_frames(self, video_path: str, num_frames: int | None = None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust frame count to match required num_frames:
        # interpolate if video has less frames, truncate if it has more
        frame_idxs = np.arange(total_frames)
        if num_frames and num_frames > total_frames:
            frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)
        elif num_frames and num_frames < total_frames:
            frame_idxs = np.arange(num_frames)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # if total_frames is less than required num_frames, duplicate some of them
            for j in range(np.count_nonzero(frame_idxs == i)):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return np.stack(frames)

    def get_embedding(self, gold_video: np.ndarray, predicted_video: np.ndarray):
        gold_inputs = self.processor(list(gold_video), return_tensors="pt")
        gold_emb = self.model(**gold_inputs).last_hidden_state[:, 0, :]

        predicted_inputs = self.processor(list(predicted_video), return_tensors="pt")
        predicted_emb = self.model(**predicted_inputs).last_hidden_state[:, 0, :]

        cos_sim_all = cosine_similarity(gold_emb.detach().numpy(), predicted_emb.detach().numpy())
        return np.mean(np.diag(cos_sim_all))

    def convert_frame_to_lpips_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        # [0, 1] -> [-1, 1]
        tensor = tensor * 2 - 1
        return tensor.unsqueeze(0)

    def get_lpips(self, gold_video: np.ndarray, pred_video: np.ndarray):
        lpips_scores = []
        for i, gold_frame in enumerate(gold_video):
            pred_frame = pred_video[i]
            pred_lpips_frame = self.convert_frame_to_lpips_tensor(pred_frame)
            gold_lpips_frame = self.convert_frame_to_lpips_tensor(gold_frame)

            with torch.no_grad():
                score = self.lpips_model(gold_lpips_frame, pred_lpips_frame)

            lpips_scores.append(score.item())

        return np.mean(lpips_scores)

    def normalize_frame_diff_for_lpips(self, diff: np.ndarray):
        # Shift and scale to [0, 255]
        diff_min = diff.min()
        diff_max = diff.max()

        if diff_max - diff_min < 1e-6:
            # No motion, return zeros
            return np.zeros_like(diff, dtype=np.uint8)

        diff_norm = (diff - diff_min) / (diff_max - diff_min) * 255
        return diff_norm.astype(np.uint8)

    def get_frame_differences(self, video: np.ndarray) -> np.ndarray:
        differences = []
        for i in range(len(video) - 1):
            diff = video[i + 1].astype(np.float32) - video[i].astype(np.float32)
            differences.append(diff)

        return np.array(differences)

    def get_temporal_lpips(self, gold_video: np.ndarray, pred_video: np.ndarray):
        """
        Temporal LPIPS: compares MOTION (frame differences) between videos
        """

        gold_video_diff = self.get_frame_differences(gold_video)
        pred_video_diff = self.get_frame_differences(pred_video)

        temporal_lpips_scores = []
        for i in range(len(gold_video_diff)):
            gold_diff_norm = self.normalize_frame_diff_for_lpips(gold_video_diff[i])
            pred_diff_norm = self.normalize_frame_diff_for_lpips(pred_video_diff[i])
            # Convert to tensor [-1, 1] range as LPIPS expects
            gold_tensor = self.convert_frame_to_lpips_tensor(gold_diff_norm)
            pred_tensor = self.convert_frame_to_lpips_tensor(pred_diff_norm)

            with torch.no_grad():
                score = self.lpips_model(gold_tensor, pred_tensor)

            temporal_lpips_scores.append(score.item())

        return np.mean(temporal_lpips_scores)

    def get_ssim(self, gold_video: np.ndarray, pred_video: np.ndarray):
        ssim_vals = []
        for i, gold_frame in enumerate(gold_video):
            gf_gray = cv2.cvtColor(gold_frame, cv2.COLOR_RGB2GRAY)
            pred_frame = pred_video[i]
            pf_gray = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2GRAY)

            ssim_vals.append(structural_similarity(gf_gray, pf_gray))

        return ssim_vals

    def evaluate(self, gt, prediction):
        videos_gold = gt["videos"].values
        videos_prediction = prediction["videos"].values

        metric_per_video = []
        ssim_per_video = []
        lpips_per_video = []
        tlpips_per_video = []
        for gold, prediction in tqdm(zip(videos_gold, videos_prediction), desc="Video Similarity evaluation"):
            # vivit requires 32 frames
            gold_video = self.load_video_frames(str(gold), num_frames=32)
            predicted_video = self.load_video_frames(str(prediction), num_frames=32)

            cos_sim_mean = self.get_embedding(gold_video, predicted_video)

            ssim_vals = self.get_ssim(gold_video, predicted_video)
            ssim_avg = sum(ssim_vals) / len(ssim_vals)

            lpips = self.get_lpips(gold_video, predicted_video)
            tlpips = self.get_temporal_lpips(gold_video, predicted_video)

            metric_per_video.append(cos_sim_mean)
            ssim_per_video.append(ssim_avg)
            lpips_per_video.append(lpips)
            tlpips_per_video.append(tlpips)

        metric_dict = {"similarity": np.mean(metric_per_video)}
        return metric_dict, {
            "similarity": metric_per_video,
            "SSIM (higher is better, 1.0 best)": ssim_per_video,
            "LPIPS (lower is better, 0 best)": lpips_per_video,
            "tLPIPS (lower is better, 0 best)": tlpips_per_video,
        }
