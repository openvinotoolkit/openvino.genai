# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import numpy as np
from PIL import Image
import logging as log
from transformers.image_utils import load_image
from .model_utils import get_param_from_file, resolve_media_file_path
from .parse_json_data import parse_text_json_data, parse_vlm_json_data, parse_image_json_data, parse_video_json_data
from pathlib import Path
import openvino as ov
import math
import cv2


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
    def inner(video_path, decim_frames, genai_flag):
        log.info(f"Input video file: {video_path}")
        if decim_frames is not None:
            log.info(f"Requested to reduce into {decim_frames} frames")
        out_frames = func(video_path, decim_frames)
        log.info(f"Final frames number: {len(out_frames)}")
        log.info(f"First frame shape: {out_frames[0].shape}")
        log.info(f"First frame dtype: {out_frames[0].dtype}")
        if genai_flag:
            return ov.Tensor(out_frames)
        return np.array(out_frames)
    return inner


@print_video_frames_number_and_convert_to_tensor
def make_video_tensor(video_path, decim_frames=None):
    assert os.path.exists(video_path), f"no input video file: {video_path}"
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

    if not decim_frames:
        log.info(f"Video decim: no-set: {decim_frames}: skip")
        return output_frames

    # decimation procedure
    # decim_frames is required max frame number if positive
    # or decimation factor if negative
    # e.g. if input frames number is 100 and decim_fames = 5:
    #         then number of processed frames are: 0, 20, 40, 60, 80
    #      if input frames number is 100 and decim_fames = -5:
    #         then number of processed frames are: 0, 5, 10, 15, 20, ...

    decim_frames = int(decim_frames)
    if decim_frames > 0:
        if len(output_frames) <= decim_frames:
            log.info(f"Video decim: too short to decim: crop: {decim_frames}")
            return list(output_frames[:decim_frames])
        decim_factor_f = float(len(output_frames)) / decim_frames
        decim_factor = int(math.ceil(decim_factor_f))
    else:
        decim_factor = -decim_frames
    log.info(f"Video decim factor: {decim_factor}")
    if decim_factor >= 2:
        return list(output_frames[::decim_factor])
    log.info("Video decim: too large decim factor: skip")
    return output_frames


def load_image_genai(image_path):
    pil_image = load_image(image_path)
    image_data = np.array(pil_image)[None]
    return ov.Tensor(image_data)


def extract_prompt_data(inputs, required_frames, genai_flag):
    prompts, images, videos = [], [], []
    if not isinstance(inputs, (list, tuple, set)):
        inputs = [inputs]
    for input_data in inputs:
        if input_data.get("video") is not None:
            entry = Path(input_data["video"])
            if entry.is_dir():
                for filename in sorted(entry.iterdir()):
                    video_tensor = make_video_tensor(filename, required_frames, genai_flag)
                    videos.append(video_tensor)
            else:
                video_tensor = make_video_tensor(entry, required_frames, genai_flag)
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


def get_vlm_prompt(args):
    vlm_file_list = []
    output_data_list, is_json_data = get_param_from_file(args, ["video", "media", "prompt"])
    if is_json_data:
        vlm_param_list = parse_vlm_json_data(output_data_list)
        if len(vlm_param_list) > 0:
            for vlm_file in vlm_param_list:
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0 and 'media' in vlm_file:
                    vlm_file['media'] = resolve_media_file_path(vlm_file.get('media'), args['prompt_file'][0])
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0 and 'video' in vlm_file:
                    vlm_file['video'] = resolve_media_file_path(vlm_file.get('video'), args['prompt_file'][0])
                vlm_file_list.append(vlm_file)
    else:
        vlm_file_list.append(output_data_list)
    return vlm_file_list


def get_image_prompt(args):
    input_image_list = []

    input_key = ["prompt"]
    if args.get("task") == args["use_case"].TASK["inpainting"]["name"] or (
        (args.get("media") or args.get("images")) and args.get("mask_image")
    ):
        input_key = ["media", "mask_image", "prompt"]
    elif args.get("task") == args["use_case"].TASK["img2img"]["name"] or args.get("media") or args.get("images"):
        input_key = ["media", "prompt"]

    output_data_list, is_json_data = get_param_from_file(args, input_key)
    if is_json_data is True:
        image_param_list = parse_image_json_data(output_data_list)
        if len(image_param_list) > 0:
            for image_data in image_param_list:
                if args["prompt_file"] is not None and len(args["prompt_file"]) > 0:
                    image_data["media"] = resolve_media_file_path(image_data.get("media"), args["prompt_file"][0])
                    image_data["mask_image"] = resolve_media_file_path(
                        image_data.get("mask_image"), args["prompt_file"][0]
                    )
                input_image_list.append(image_data)
    else:
        input_image_list.append(output_data_list[0])
    return input_image_list


def get_video_gen_prompt(args):
    input_list = []
    output_data_list, is_json_data = get_param_from_file(args, ["prompt", "negative_prompt"])
    if is_json_data is True:
        media_param_list = parse_video_json_data(output_data_list)
        if len(media_param_list) > 0:
            for text in media_param_list:
                input_list.append(text)
    else:
        input_list.append(output_data_list[0])
    return input_list


#########################################################
# TODO: implement BenchPrompt and BenchPrompter
#
# CLASS BenchPrompt is handler for a single multimedia prompt
#   - it load prompt content/data: text, image, video, and/or audio
#   - data should be store inside under keys: prompt, image, video, audio, mask, etc.
#   - it should evaluate correctnes of input data (decimate/reduce if needed especially for video see make_video_tensor)
#   - it should be able to create prompt representation (repr) with size of its elements:
#     -- text -> <text length in tokens>
#     -- text + image -> <text length in tokens> + <width in pixels>x<height in pixels>
#     -- text + video -> <text length in tokens> + <width in pixels>x<height in pixels>@<frames>
#     -- text + audio -> <text length in tokens> + <duration in seconds>@<sampling-rate>
#     -- text + image + mask -> <text length in tokens> + <width in pixels>x<height in pixels>/<fraction in percentege>%
#   - please note that prompts can be loaded directly from command line of from jsonl files e.g.:
#     -- example 1: 
#     {"type": "audio", "media": "./audio/intel_ad_30s_124kbps.mp3"}
#     {"type": "audio", "media": "./audio/intel_ad_90s_128kbps.mp3"}
#     -- example 2:
#     {"prompt": "What is presented in the video?", "video": "./video/spinning-earth-480.mp4"}
#     {"prompt": "What is Earth's spin and which continents are visible over time in the video? Which of them are shown on the beginning and which of them are presented on the end of the clip?", "video": "./video/spinning-earth-480.mp4"}
#     -- example 3:
#     {"prompt": "What is in the image?", "media": "./image/overture-creations-5sI6fQgYIuo.png"}
#     -- example 4:
#      {"word_num": "32", "type": "image_description", "language": "EN", "source": "Gustavosta/Stable-Diffusion-Prompts", "prompt": "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, beautifully backlit, swirly vibrant color lines, fantastically gaudy, aesthetic octane render, 8K HD Resolution"}
#   - see tools/llm_bench/benchmark.py for possible options how to provide prompts
#
# CLASS BenchPrompter is a container for multiple BenchPrompts
#   - it parses and analyzes command line in context of prompts loading
#   - prompt index should be possition in BenchPrompter which is list
#   - it gives an iterator for main loops with proper sequence of prompts
#     with respect of: args.prompt_index, args.batch_size, args.streaming, args.subsequent, etc.
#
# TODO: refactor code in tools/llm_bench/task/text_generation.py to use new classes
#     in tools/llm_bench/task/text_generation.py in run_text_generation_benchmark
#     it seems that code under: "if args['subsequent'] is False:" is very similar with code under "else"
#     propose unification of this code mapping idx -> p_idx inside BenchPrompter class
#

class BenchPrompt(dict):
    def __init__(self, args):
        dict.__init__(self)


class BenchPrompter(list):
    def __init__(self, args):
        list.__init__(self)

    def append(self, prompt):
        assert type(prompt) is BenchPrompt
