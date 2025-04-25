# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino
import openvino_genai as ov_genai
import numpy as np
from PIL import Image

def get_total_text_encoder_infer_duration(metrics):
    total_duration = 0.0
    for key, value in metrics.get_text_encoder_infer_duration().items():
        total_duration = total_duration + value
    return total_duration

def print_one_generate(metrics, prefix, idx):
    prefix_idx = "[" + prefix + "-" + str(idx) + "]"
    print(f"\n{prefix_idx} generate time: {metrics.get_generate_duration():.2f} ms, total infer time: {metrics.get_inference_duration():.2f} ms")
    print(f"{prefix_idx} text encoder infer time: {get_total_text_encoder_infer_duration(metrics):.2f} ms")
    first_iter_time = 0.0
    other_iter_avg_time = 0.0
    first_infer_time = 0.0
    other_infer_avg_time = 0.0
    first_iter_time, other_iter_avg_time = metrics.get_first_and_other_iter_duration()
    if len(metrics.raw_metrics.transformer_inference_durations) > 0:
        first_infer_time, other_infer_avg_time = metrics.get_first_and_other_trans_infer_duration()
        print(f"{prefix_idx} transformer iteration num: {len(metrics.raw_metrics.iteration_durations)}, first iteration time: {first_iter_time:.2f} ms, other iteration avg time: {other_iter_avg_time:.2f} ms")
        print(f"{prefix_idx} transformer inference num: {len(metrics.raw_metrics.transformer_inference_durations)}, first inference time: {first_infer_time:.2f} ms, other inference avg time: {other_infer_avg_time:.2f} ms")
    else:
        first_infer_time, other_infer_avg_time = metrics.get_first_and_other_unet_infer_duration()
        print(f"{prefix_idx} unet iteration num: {len(metrics.raw_metrics.iteration_durations)}, first iteration time: {first_iter_time:.2f} ms, other iteration avg time: {other_iter_avg_time:.2f} ms")
        print(f"{prefix_idx} unet inference num: {len(metrics.raw_metrics.unet_inference_durations)}, first inference time: {first_infer_time:.2f} ms, other inference avg time: {other_infer_avg_time:.2f} ms")
    print(f"{prefix_idx} vae encoder infer time: {metrics.get_vae_encoder_infer_duration():.2f} ms, vae decoder infer time: {metrics.get_vae_decoder_infer_duration():.2f} ms")
    
def print_statistic(warmup_metrics, iter_metrics):
    generate_durations = []
    inference_durations = []
    text_encoder_durations = []
    vae_encoder_durations = []
    vae_decoder_durations = []
    load_time = 0.0
    warmup_num = len(warmup_metrics)
    iter_num = len(iter_metrics)
    generate_warmup = 0.0
    inference_warmup = 0.0
    if warmup_num > 0:
        generate_warmup = warmup_metrics[0].get_generate_duration()
        inference_warmup = warmup_metrics[0].get_inference_duration()

    for metrics in iter_metrics:
        generate_durations.append(metrics.get_generate_duration())
        inference_durations.append(metrics.get_inference_duration())
        text_encoder_durations.append(get_total_text_encoder_infer_duration(metrics))
        vae_encoder_durations.append(metrics.get_vae_encoder_infer_duration())
        vae_decoder_durations.append(metrics.get_vae_decoder_infer_duration())
        load_time = metrics.get_load_time()
        
    generate_mean = sum(generate_durations)
    if (len(generate_durations) > 0):
        generate_mean = generate_mean / len(generate_durations)
        
    inference_mean = sum(inference_durations)
    if (len(inference_durations) > 0):
        inference_mean = inference_mean / len(inference_durations)
        
    text_encoder_mean = sum(text_encoder_durations)
    if (len(text_encoder_durations) > 0):
        text_encoder_mean = text_encoder_mean / len(text_encoder_durations)

    vae_encoder_mean = sum(vae_encoder_durations)
    if (len(vae_encoder_durations) > 0):
        vae_encoder_mean = vae_encoder_mean / len(vae_encoder_durations)
        
    vae_decoder_mean = sum(vae_decoder_durations)
    if (len(vae_decoder_durations) > 0):
        vae_decoder_mean = vae_decoder_mean / len(vae_decoder_durations)
        
    print(f"\nTest finish, load time: {load_time:.2f} ms")
    print(f"Warmup number: {warmup_num}, first generate warmup time: {generate_warmup:.2f} ms, infer warmup time: {inference_warmup:.2f} ms")
    print(f"Generate iteration number: {iter_num}, for one iteration, generate avg time: {generate_mean:.2f} ms, "
          f"infer avg time: {inference_mean:.2f} ms, all text encoder infer avg time: {text_encoder_mean:.2f} ms, "
          f"vae encoder infer avg time: {vae_encoder_mean:.2f} ms, vae decoder infer avg time: {vae_decoder_mean:.2f} ms")

def device_string_to_triplet(device_input):
    devices = [device.strip() for device in device_input.split(",")]
    if len(devices) == 1:
        return [devices[0]] * 3
    elif len(devices) == 3:
        return devices
    else:
        raise ValueError("The device specified by -d/--device must be a single device (e.g. -d \"GPU\"), " +
                         "or exactly 3 comma separated device names (e.g. -d \"CPU,NPU,GPU\")")

def text2image(args):
    prompt = args.prompt
    models_path = args.model
    devices = device_string_to_triplet(args.device)
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    
    pipe = ov_genai.Text2ImagePipeline(models_path)
    if args.reshape:
        pipe.reshape(args.num_images_per_prompt, args.height, args.width, pipe.get_generation_config().guidance_scale)
    pipe.compile(devices[0], devices[1], devices[2])

    config = pipe.get_generation_config()
    config.width = args.width
    config.height = args.height
    config.num_inference_steps = args.num_inference_steps
    config.num_images_per_prompt = args.num_images_per_prompt
    pipe.set_generation_config(config)
    
    warmup_metrics = []
    for i in range(num_warmup):
        pipe.generate(prompt)
        metrics = pipe.get_performance_metrics()
        warmup_metrics.append(metrics)
        print_one_generate(metrics, "warmup", i)
    
    iter_metrics = []
    for i in range(num_iter):
        image_tensor = pipe.generate(prompt)
        perf_metrics = pipe.get_performance_metrics()
        iter_metrics.append(perf_metrics)
        image = Image.fromarray(image_tensor.data[0])
        image_name = output_dir + "/image_" + str(i) + ".bmp"
        image.save(image_name)
        print_one_generate(perf_metrics, "iter", i)
        
    print_statistic(warmup_metrics, iter_metrics)

def read_image(path: str) -> openvino.Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)[None]
    return openvino.Tensor(image_data)

def image2image(args):
    prompt = args.prompt
    models_path = args.model
    devices = device_string_to_triplet(args.device)
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    image_path = args.image
    strength = args.strength
    
    image_input = read_image(image_path)

    pipe = ov_genai.Image2ImagePipeline(models_path)
    if args.reshape:
        height = image_input.get_shape()[1]
        width = image_input.get_shape()[2]
        pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)
    pipe.compile(devices[0], devices[1], devices[2])

    warmup_metrics = []
    for i in range(num_warmup):
        pipe.generate(prompt, image_input, strength=strength)
        metrics = pipe.get_performance_metrics()
        warmup_metrics.append(metrics)
        print_one_generate(metrics, "warmup", i)
    
    iter_metrics = []
    for i in range(num_iter):
        image_tensor = pipe.generate(prompt, image_input, strength=strength)
        perf_metrics = pipe.get_performance_metrics()
        iter_metrics.append(perf_metrics)
        image = Image.fromarray(image_tensor.data[0])
        image_name = output_dir + "/image_" + str(i) + ".bmp"
        image.save(image_name)
        print_one_generate(perf_metrics, "iter", i)
        
    print_statistic(warmup_metrics, iter_metrics)

def inpainting(args):
    prompt = args.prompt
    models_path = args.model
    devices = device_string_to_triplet(args.device)
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    image_path = args.image
    strength = args.strength
    mask_image_path = args.mask_image
    
    image_input = read_image(image_path)
    mask_image = read_image(mask_image_path)

    pipe = ov_genai.InpaintingPipeline(models_path)
    if args.reshape:
        height = image_input.get_shape()[1]
        width = image_input.get_shape()[2]
        pipe.reshape(1, height, width, pipe.get_generation_config().guidance_scale)
    pipe.compile(devices[0], devices[1], devices[2])

    warmup_metrics = []
    for i in range(num_warmup):
        pipe.generate(prompt, image_input, mask_image)
        metrics = pipe.get_performance_metrics()
        warmup_metrics.append(metrics)
        print_one_generate(metrics, "warmup", i)
    
    iter_metrics = []
    for i in range(num_iter):
        image_tensor = pipe.generate(prompt, image_input, mask_image)
        perf_metrics = pipe.get_performance_metrics()
        iter_metrics.append(perf_metrics)
        image = Image.fromarray(image_tensor.data[0])
        image_name = output_dir + "/image_" + str(i) + ".bmp"
        image.save(image_name)
        print_one_generate(perf_metrics, "iter", i)
        
    print_statistic(warmup_metrics, iter_metrics)
    

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-t", "--pipeline_type", type=str, default="text2image", help="pipeline type: text2image/image2image/inpainting")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=3, help="Number of iterations")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Path to save output image")
    parser.add_argument("-is", "--num_inference_steps", type=int, default=20, help="The number of inference steps used to denoise initial noised latent to final image")
    parser.add_argument("-ni", "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per generate() call")
    parser.add_argument("-i", "--image", type=str, help="Image path")
    parser.add_argument("-r", "--reshape", action="store_true", help="Reshape pipeline before compilation")
    # special parameters of text2image pipeline
    parser.add_argument("-w", "--width", type=int, default=512, help="The width of the resulting image")
    parser.add_argument("-ht", "--height", type=int, default=512, help="The height of the resulting image")
    # special parameters of image2image pipeline
    parser.add_argument("-s", "--strength", type=float, default=0.8, help="Indicates extent to transform the reference `image`. Must be between 0 and 1")
    # special parameters of inpainting pipeline
    parser.add_argument("-mi", "--mask_image", type=str, help="Mask image path")
    
    args = parser.parse_args()
    
    type = args.pipeline_type
    
    if type == "text2image":
        text2image(args)
    elif type == "image2image":
        image2image(args)
    elif type == "inpainting":
        inpainting(args)
    else:
        print(f"not support pipeline type: {type}\n")

if __name__ == "__main__":
    main()
