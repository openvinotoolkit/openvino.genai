# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import cv2

import numpy as np
import soundfile as sf


def get_file_path(output_dir, file_name):
    if output_dir is not None:
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
        out_path = output_dir
    else:
        out_path = '.'

    return out_path + os.sep + file_name


def save_text_to_file(input_text, text_file_name, args):
    save_path = get_file_path(args["output_dir"], text_file_name)
    input_text_file = open(save_path, "w")
    input_text_file.write(input_text)
    input_text_file.close()


def save_image_file(img, img_file_name, args):
    save_path = get_file_path(args["output_dir"], img_file_name)
    img.save(save_path)
    return save_path


def save_audio_file(audio, audio_file_name, args, samplerate=16000):
    save_path = get_file_path(args["output_dir"], audio_file_name)
    sf.write(save_path, audio, samplerate)
    return save_path


def save_video_file(
    frames,
    video_save_name,
    args,
    fps: int,
):
    save_path = get_file_path(args["output_dir"], video_save_name)

    w, h = frames[0].size[0], frames[0].size[1]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    for frame in frames:
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def construct_file_name(batch_size, model_name, model_precision, prompt_idx, iteration, batchsize_idx, proc_id, suffix):
    file_save_name = model_name
    if model_precision:
        file_save_name += "_" + model_precision
    file_save_name += "_p" + str(prompt_idx)
    if batch_size > 1 and batchsize_idx is not None:
        file_save_name += "_bs" + str(batchsize_idx)
    if iteration:
        file_save_name += "_iter" + str(iteration)
    file_save_name += "_pid" + str(proc_id) + suffix
    return file_save_name


def output_input_text(input_text, args, model_precision, prompt_idx, batchsize_idx, proc_id):
    text_file_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision,
        prompt_idx,
        iteration=None,
        batchsize_idx=batchsize_idx,
        proc_id=proc_id,
        suffix="_input.txt",
    )
    save_text_to_file(input_text, text_file_name, args)


def output_image_input_text(input_text, args, prompt_idx, batchsize_idx, proc_id):
    text_file_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision=None,
        prompt_idx=prompt_idx,
        iteration=None,
        batchsize_idx=batchsize_idx,
        proc_id=proc_id,
        suffix="_input.txt",
    )
    save_text_to_file(input_text, text_file_name, args)


def output_gen_text(generated_text, args, model_precision, prompt_idx, iteration, batchsize_idx, proc_id):
    text_file_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision,
        prompt_idx,
        iteration,
        batchsize_idx,
        proc_id=proc_id,
        suffix="_output.txt",
    )
    save_text_to_file(generated_text, text_file_name, args)


def output_gen_image(img, args, prompt_idx, iteration, batchsize_idx, proc_id, suffix):
    img_save_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision=None,
        prompt_idx=prompt_idx,
        iteration=iteration,
        batchsize_idx=batchsize_idx,
        proc_id=proc_id,
        suffix=f"_output{suffix}",
    )
    img_save_path = save_image_file(img, img_save_name, args)
    return img_save_path


def output_gen_audio(audio, args, prompt_idx, iteration, batchsize_idx, proc_id, suffix):
    audio_save_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision=None,
        prompt_idx=prompt_idx,
        iteration=iteration,
        batchsize_idx=batchsize_idx,
        proc_id=proc_id,
        suffix=f"_output{suffix}",
    )
    audio_save_path = save_audio_file(audio, audio_save_name, args)
    return audio_save_path


def output_gen_video(video, args, prompt_idx, iteration, batchsize_idx, proc_id, suffix, fps: int):
    video_save_name = construct_file_name(
        args["batch_size"],
        args["model_name"],
        model_precision=None,
        prompt_idx=prompt_idx,
        iteration=iteration,
        batchsize_idx=batchsize_idx,
        proc_id=proc_id,
        suffix=f"_output{suffix}",
    )
    video_save_path = save_video_file(video, video_save_name, args, fps)
    return video_save_path
