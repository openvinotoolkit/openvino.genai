# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import cv2
import numpy as np
from PIL import Image
import logging as log
from transformers.image_utils import load_image
from .model_utils import get_param_from_file
from .model_utils import resolve_media_file_path
from .parse_json_data import parse_text_json_data
from .parse_json_data import parse_vlm_json_data
from pathlib import Path
import openvino as ov


def get_text_prompt(args):
    text_list = []
    output_data_list, is_json_data = get_param_from_file(args, 'prompt')
    if is_json_data is True:
        text_param_list = parse_text_json_data(output_data_list)
        if len(text_param_list) > 0:
            for text in text_param_list:
                text_list.append(text)
    else:
        text_list.append(output_data_list[0])
    return text_list


def print_video_frames_number_and_convert_to_tensor(func):
    def inner(video_path, decym_frames, genai_flag):
        log.info(f"Input video file: {video_path}")
        if decym_frames is not None:
            log.info(f"Requested to reduce into {decym_frames} frames")
        out_frames = func(video_path, decym_frames)
        log.info(f"Final frames number: {len(out_frames)}")
        log.info(f"First frame shape: {out_frames[0].shape}")
        log.info(f"First frame dtype: {out_frames[0].dtype}")
        if genai_flag:
            return [ov.Tensor(frame) for frame in out_frames]
        return np.array(out_frames)
    return inner


@print_video_frames_number_and_convert_to_tensor
def make_video_tensor(video_path, decym_frames=None):
    supported_files = {
        '.mp4',   # MPEG-4 (most common)
        '.avi',   # Audio Video Interleave
        '.mov',   # QuickTime Movie
        '.mkv',   # Matroska Video
        '.wmv',   # Windows Media Video
        '.flv',   # Flash Video
        '.webm',  # WebM
        '.m4v',   # iTunes Video
        '.3gp',   # 3GPP
        '.mpeg',  # MPEG
        '.mpg'    # MPEG
    }

    assert os.path.exists(video_path), f"no input video file: {video_path}"
    assert video_path.suffix.lower() in supported_files, "no supported video file"
    cap = cv2.VideoCapture(video_path)

    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        np_img_array = np.array(pil_image)
        log.debug(f"Video shape: {np_img_array.shape}")
        log.debug(f"Video dtype: {np_img_array.dtype}")
        output_frames.append(np_img_array)

    if decym_frames is None:
        log.info("Video decym: none: skip")
        return output_frames
    if int(decym_frames) == 0:
        log.info("Video decym: zero: skip")
        return output_frames

    # decymation procedure
    # decym_fames is required max frame number if positive
    # or decymation factor if negative

    decym_frames = int(decym_frames)
    if decym_frames > 0:
        if len(output_frames) <= decym_frames:
            log.info(f"Video decym: too short to decym: crop: {decym_frames}")
            return list(output_frames[:decym_frames])
        decym_factor = 1 + int(len(output_frames) / decym_frames)
    else:
        decym_factor = -decym_frames
    log.info(f"Video decym factor: {decym_factor}")
    if decym_factor >= 2:
        return list(output_frames[::decym_factor])
    log.info("Video decym: too large decym factor: skip")
    return output_frames


def load_image_genai(image_path):
    pil_image = load_image(image_path)
    image_data = np.array(pil_image)[None]
    return ov.Tensor(image_data)


def extract_prompt_issues(inputs, required_frames, genai_flag):
    prompts, images, videos = [], [], []
    if not isinstance(inputs, (list, tuple, set)):
        inputs = [inputs]
    for input_data in inputs:
        if input_data.get("video") is not None:
            entry = Path(input_data["video"])
            if entry.is_dir():
                for filename in sorted(entry.iterdir()):
                    video_tensor = make_video_tensor(filename, required_frames, genai_flag)
                    if genai_flag:
                        videos.extend(video_tensor)
                    else:
                        videos.append(video_tensor)
            else:
                video_tensor = make_video_tensor(entry, required_frames, genai_flag)
                if genai_flag:
                    videos.extend(video_tensor)
                else:
                    videos.append(video_tensor)
        if input_data.get("media") is not None:
            func_load_image = load_image_genai if genai_flag else load_image
            entry = Path(input_data["media"])
            if entry.is_dir():
                for file in sorted(entry.iterdir()):
                    img = func_load_image(str(file))
                    images.append(img)
            else:
                img = func_load_image(input_data["media"])
                images.append(img)
        prompts.append(input_data["prompt"])
    return prompts, images, videos


def get_image_text_prompt(args):
    vlm_file_list = []
    output_data_list, is_json_data = get_param_from_file(args, ["video", "media", "prompt"])
    if is_json_data:
        vlm_param_list = parse_vlm_json_data(output_data_list)
        if len(vlm_param_list) > 0:
            for vlm_file in vlm_param_list:
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0 and 'media' in vlm_file:
                    if 'video' in vlm_file:
                        raise ValueError('media and video cannot be specify in a single prompt file')
                    vlm_file['media'] = resolve_media_file_path(vlm_file.get('media'), args['prompt_file'][0])
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0 and 'video' in vlm_file:
                    vlm_file['video'] = resolve_media_file_path(vlm_file.get('video'), args['prompt_file'][0])
                vlm_file_list.append(vlm_file)
    else:
        vlm_file_list.append(output_data_list)
    return vlm_file_list
