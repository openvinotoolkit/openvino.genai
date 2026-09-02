# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def create_base_prompt(json_data, key="prompt", optional=False):
    prompt_data = {}
    if key not in json_data:
        if optional:
            prompt_data[key] = ""
            return prompt_data
        raise RuntimeError(f"== key word '{key}' does not exist ==")
    if json_data[key] == "" and not optional:
        raise RuntimeError(f"== {key} should not be empty string ==")
    prompt_data[key] = json_data[key]
    return prompt_data


def parse_text_json_data(json_data_list):
    text_param_list = []
    for json_data in json_data_list:
        prompt_data = create_base_prompt(json_data)
        text_param_list.append(prompt_data["prompt"])
    return text_param_list


def parse_vlm_json_data(json_data_list, optional_prompt=False):
    text_param_list = []
    for json_data in json_data_list:
        data_list = json_data
        chat_mode = isinstance(json_data, list)
        if not chat_mode:
            data_list = [json_data]

        new_data = []
        for data in data_list:
            prompt_data = create_base_prompt(data, optional=optional_prompt)
            for param in ["media", "video", "audio"]:
                if param in data:
                    prompt_data[param] = data[param]
            new_data.append(prompt_data)

        if chat_mode:
            text_param_list.append(new_data)
        else:
            text_param_list.extend(new_data)

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
        if "prompt" in json_data:
            speech_param["prompt"] = json_data["prompt"]
        speech_param_list.append(speech_param)
    return speech_param_list
