# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino
import openvino_genai as ov_genai
from PIL import Image

def print_one_generate(metrics, prefix, idx):
    prefix_idx = "[" + prefix + "-" + str(idx) + "]"
    print(f"\n{prefix_idx} generate time: {metrics.get_generate_duration()} ms, total infer time: {metrics.get_inference_total_duration()} ms\n")
    print(f"{prefix_idx} encoder infer time: {metrics.get_encoder_infer_duration()} ms\n")
    first_iter_time = 0.0
    other_iter_avg_time = 0.0
    first_infer_time = 0.0
    other_infer_avg_time = 0.0
    metrics.get_iteration_duration(first_iter_time, other_iter_avg_time)
    if not metrics.raw_metrics.transformer_inference_durations:
        metrics.get_transformer_infer_duration(first_infer_time, other_infer_avg_time)
        print(f"{prefix_idx} transformer iteration num: {len(metrics.raw_metrics.iteration_durations)}, first iteration time: {first_iter_time} ms, other iteration avg time: {other_iter_avg_time} ms\n")
        print(f"{prefix_idx} transformer inference num: {len(metrics.raw_metrics.transformer_inference_durations)}, first inference time: {first_infer_time} ms, other inference avg time: {other_infer_avg_time} ms\n")
    else:
        metrics.get_unet_infer_duration(first_infer_time, other_infer_avg_time)
        print(f"{prefix_idx} unet iteration num: {len(metrics.raw_metrics.iteration_durations)}, first iteration time: {first_iter_time} ms, other iteration avg time: {other_iter_avg_time} ms\n")
        print(f"{prefix_idx} unet inference num: {len(metrics.raw_metrics.unet_inference_durations)}, first inference time: {first_infer_time} ms, other inference avg time: {other_infer_avg_time} ms\n")
    print(f"{prefix_idx} vae decoder infer time: {metrics.vae_decoder_inference_duration} ms\n")
    
def print_statistic(warmup_metrics, iter_metrics):
    generate_durations = []
    inference_durations = []
    encoder_durations = []
    decoder_durations = []
    load_time = 0.0
    warmup_num = len(warmup_metrics)
    iter_num = len(iter_metrics)
    generate_warmup = 0.0
    inference_warmup = 0.0
    if warmup_num > 0:
        generate_warmup = warmup_metrics[0].get_generate_duration()
        inference_warmup = warmup_metrics[0].get_all_infer_duration()

    for metrics in iter_metrics:
        generate_durations.append(metrics.get_generate_duration())
        inference_durations.append(metrics.get_all_infer_duration())
        encoder_durations.append(metrics.get_encoder_infer_duration())
        decoder_durations.append(metrics.get_decoder_infer_duration())
        load_time = metrics.get_load_time()
        
    generate_mean = sum(generate_durations)
    if (len(generate_durations) > 0):
        generate_mean = generate_mean / len(generate_durations)
        
    inference_mean = sum(inference_durations)
    if (len(inference_durations) > 0):
        inference_mean = inference_mean / len(inference_durations)

    encoder_mean = sum(encoder_durations)
    if (len(encoder_durations) > 0):
        encoder_mean = encoder_mean / len(encoder_durations)
        
    decoder_mean = sum(decoder_durations)
    if (len(decoder_durations) > 0):
        decoder_mean = decoder_mean / len(decoder_durations)
        
    print(f"\nTest finish, load time: {load_time} ms\n")
    print(f"Warmup number: {warmup_num}, first generate warmup time: {generate_warmup} ms, infer warmup time: {inference_warmup} ms\n")
    print(f"Generate iteration number: {iter_num}, for one iteration, generate avg time: {generate_mean} ms, infer avg time: {inference_mean} ms, total encoder infer avg time: {encoder_mean} ms, decoder infer avg time: {decoder_mean} ms\n")

def text2image(args):
    prompt = args.prompt
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    
    pipe = ov_genai.Text2ImagePipeline(models_path, device)
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
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return openvino.Tensor(image_data)

def image2image(args):
    prompt = args.prompt
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    image_path = args.image
    strength = args.strength
    
    pipe = ov_genai.Image2ImagePipeline(models_path, device)

    image_input = read_image(image_path)

    warmup_metrics = []
    for i in range(num_warmup):
        pipe.generate(prompt, image_input, strength)
        metrics = pipe.get_performance_metrics()
        warmup_metrics.append(metrics)
        print_one_generate(metrics, "warmup", i)
    
    iter_metrics = []
    for i in range(num_iter):
        image_tensor = pipe.generate(prompt, image_input, strength)
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
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    output_dir = args.output_dir
    image_path = args.image
    strength = args.strength
    mask_image_path = args.mask_image
    
    pipe = ov_genai.InpaintingPipeline(models_path, device)

    image_input = read_image(image_path)
    mask_image = read_image(mask_image_path)

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
    parser.add_argument("-pt", "--pipeline_type", type=str, default="text2image", help="pipeline type: text2image/image2image/inpainting")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Path to save output image")
    parser.add_argument("-is", "--num_inference_steps", type=int, default=20, help="The number of inference steps used to denoise initial noised latent to final image")
    parser.add_argument("-ni", "--num_images_per_prompt", type=int, default=512, help="The number of images to generate per generate() call")
    parser.add_argument("-i", "--image", type=str, help="Image path")
    # special parameters of text2image pipeline
    parser.add_argument("-wh", "--width", type=int, default=512, help="The width of the resulting image")
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
