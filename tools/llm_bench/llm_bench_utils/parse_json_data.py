# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def create_base_prompt(json_data, key='prompt'):
    prompt_data = {}
    if key not in json_data:
        raise RuntimeError(f"== key word '{key}' does not exist ==")
    if json_data[key] == "":
        raise RuntimeError(f"== {key} should not be empty string ==")
    prompt_data[key] = json_data[key]
    return prompt_data


def parse_text_json_data(json_data_list):
    text_param_list = []
    for json_data in json_data_list:
        prompt_data = create_base_prompt(json_data)
        text_param_list.append(prompt_data["prompt"])
    return text_param_list


def parse_vlm_json_data(json_data_list):
    text_param_list = []
    for json_data in json_data_list:
        prompt_data = create_base_prompt(json_data)
        for param in ["media", "video"]:
            if param in json_data:
                prompt_data[param] = json_data[param]

        text_param_list.append(prompt_data)
    return text_param_list


def parse_image_json_data(json_data_list):
    image_param_list = []
    for json_data in json_data_list:
        image_param = create_base_prompt(json_data)
        for param in ["width", "height", "steps"]:
            if param in json_data:
                image_param[param] = int(json_data[param])

        for param in ["media", "mask_image"]:
            if param in json_data:
                image_param[param] = json_data[param]

        if 'guidance_scale' in json_data:
            image_param['guidance_scale'] = float(json_data['guidance_scale'])

        image_param_list.append(image_param)
    return image_param_list


def parse_video_json_data(json_data_list):
    video_param_list = []
    for json_data in json_data_list:
        video_param = create_base_prompt(json_data)
        for param in ["width", "height", "num_steps", "num_frames", "frame_rate"]:
            if param in json_data:
                video_param[param] = int(json_data[param])

        for param in ["guidance_scale", "guidance_rescale"]:
            if param in json_data:
                video_param[param] = float(json_data[param])

        video_param_list.append(video_param)
    return video_param_list


def parse_speech_json_data(json_data_list):
    speech_param_list = []
    for json_data in json_data_list:
        speech_param = create_base_prompt(json_data, "media")
        if "language" in json_data:
            speech_param["language"] = json_data["language"]
        if "timestamp" in json_data:
            speech_param["timestamp"] = json_data["timestamp"]
        speech_param_list.append(speech_param)
    return speech_param_list
