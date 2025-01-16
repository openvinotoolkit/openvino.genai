# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai as ov_genai
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Path to save output image")
    parser.add_argument("-wh", "--width", type=int, default=512, help="The width of the resulting image")
    parser.add_argument("-ht", "--height", type=int, default=512, help="The height of the resulting image")
    parser.add_argument("-is", "--num_inference_steps", type=int, default=20, help="The number of inference steps used to denoise initial noised latent to final image")
    parser.add_argument("-ni", "--num_images_per_prompt", type=int, default=512, help="The number of images to generate per generate() call")
    
    args = parser.parse_args()

    # Perf metrics is stored in DecodedResults. 
    # In order to get DecodedResults instead of a string input should be a list.
    prompt = [args.prompt]
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
    
    for _ in range(num_warmup):
        pipe.generate(prompt)
    
    generate_durations = []
    total_inference_durations = []
    load_time = 0
    for i in range(num_iter):
        image_tensor = pipe.generate(prompt)
        perf_metrics = pipe.get_performance_metrics()
        generate_durations.append(perf_metrics.get_generate_duration())
        total_inference_durations.append(perf_metrics.get_inference_total_duration())
        image = Image.fromarray(image_tensor.data[0])
        image_name = output_dir + "/image_" + str(i) + ".bmp"
        image.save(image_name)
        load_time = perf_metrics.get_load_time()
        
    generate_mean = sum(generate_durations)
    if (len(generate_durations) > 0):
        generate_mean = generate_mean / len(generate_durations)
        
    inference_mean = sum(total_inference_durations)
    if (len(total_inference_durations) > 0):
        inference_mean = inference_mean / len(total_inference_durations)
    
    print(f"Load time: {load_time:.2f} ms")
    print(f"One generate avg time:: {generate_mean:.2f} ms")
    print(f"Total inference for one generate avg time: {inference_mean:.2f} ms")

if __name__ == "__main__":
    main()
