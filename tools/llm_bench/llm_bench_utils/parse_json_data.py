# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
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
        if "media" in json_data:
            prompt_data["media"] = json_data["media"]
        if "video" in json_data:
            prompt_data["video"] = json_data["video"]
        text_param_list.append(prompt_data)
    return text_param_list


def parse_image_json_data(json_data_list):
    image_param_list = []
    for json_data in json_data_list:
        image_param = create_base_prompt(json_data)
        if 'width' in json_data:
            image_param['width'] = int(json_data['width'])
        if 'height' in json_data:
            image_param['height'] = int(json_data['height'])
        if 'steps' in json_data:
            image_param['steps'] = int(json_data['steps'])
        if 'guidance_scale' in json_data:
            image_param['guidance_scale'] = float(json_data['guidance_scale'])
        if 'media' in json_data:
            image_param['media'] = json_data['media']
        if 'mask_image' in json_data:
            image_param['mask_image'] = json_data['mask_image']
        image_param_list.append(image_param)
    return image_param_list


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
