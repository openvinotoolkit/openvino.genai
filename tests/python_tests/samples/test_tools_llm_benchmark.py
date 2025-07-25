# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, convert_model, download_test_content
from test_utils import run_sample

convert_draft_model = convert_model
download_mask_image = download_test_content

image_generation_prompt = "side profile centered painted portrait, Gandhi rolling a blunt, Gloomhaven, matte painting concept art, art nouveau, 8K HD Resolution, beautifully background"
image_generation_json = [
    {"steps": 30, "width": 64, "height": 128, "guidance_scale": 1.0, "prompt": image_generation_prompt},
    {"steps": 4, "width": 64, "height": 32, "guidance_scale": 7.0, "prompt": image_generation_prompt}
]
image_generation_inpainting_json = [
    {"steps": 30, "width": 64, "height": 128, "guidance_scale": 1.0, "strength": "0.8", "media": "overture-creations.png", "mask_image": "overture-creations-mask.png", "prompt": image_generation_prompt},
]
image_generation_i2i_prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image_generation_i2i_json = [
    {"steps": 30, "width": 64, "height": 128, "guidance_scale": 1.0, "strength": "0.8", "media": "cat.png", "prompt": image_generation_i2i_prompt},
]

class TestBenchmarkLLM:
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "download_model, sample_args",
        [
            pytest.param("tiny-dummy-qwen2", ["-d", "cpu", "-n", "1", "-f", "pt", "-ic", "20"]),
        ],
        indirect=["download_model"],
    )
    def test_python_tool_llm_benchmark_download_model(self, download_model, sample_args):
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , download_model] + sample_args
        run_sample(benchmark_py_command)
        
        
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-qwen2", ["-d", "cpu", "-n", "1", "-ic", "10", "--optimum"]),
            pytest.param("tiny-random-qwen2", ["-d", "cpu", "-n", "1", "-ic", "10", "--optimum", "--num_beams", "2"]),
            pytest.param("tiny-random-qwen2", ["-d", "cpu", "-n", "1", "-ic", "20", "--max_ngram_size", "3", "--num_assistant_tokens", "5", "-p", "'Why is the Sun yellow?'"]),
            pytest.param("tiny-random-llava", [ "-ic", "4", "-pf", os.path.join(SAMPLES_PY_DIR, "llm_bench/prompts/llava-1.5-7b.jsonl")]),
            pytest.param("tiny-random-llava", [ "-ic", "4", "--optimum", "-pf", os.path.join(SAMPLES_PY_DIR, "llm_bench/prompts/llava-1.5-7b.jsonl")]),
            pytest.param("tiny-random-latent-consistency", [ "-d", "cpu", "-n", "1", "--num_steps", "4", "--static_reshape", "-p", "'an astronaut riding a horse on mars'"]),
            pytest.param("tiny-random-latent-consistency", [ "-d", "cpu", "-n", "1", "--num_steps", "4", "--static_reshape", "-p", "'an astronaut riding a horse on mars'", "--optimum"]),
        ],
        indirect=["convert_model"],
    )
    def test_python_tool_llm_benchmark_convert_model(self, convert_model, sample_args):
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model] + sample_args
        run_sample(benchmark_py_command)       
        
        
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-llava", [ "-ic", "20", "--prompt", "'What is unusual on this image?'"]),
            pytest.param("tiny-random-llava", [ "-ic", "20", "--optimum", "--prompt", "'What is unusual on this image?'"]),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["cat"], indirect=True)
    def test_python_tool_llm_benchmark_convert_model_media(self, convert_model, download_test_content, sample_args):
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model, "--media", download_test_content] + sample_args
        run_sample(benchmark_py_command)      


    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, convert_draft_model, sample_args",
        [
            pytest.param("tiny-random-qwen2", "tiny-random-qwen2-int8", ["-d", "cpu", "--draft_device", "cpu", "-n", "1", "--assistant_confidence_threshold", "0.4", "-ic", "20"]),
            pytest.param("tiny-random-qwen2", "tiny-random-qwen2-int8", ["-d", "cpu", "--draft_device", "cpu", "-n", "1", "--num_assistant_tokens", "5", "-ic", "20"]),
        ],
        indirect=["convert_model", "convert_draft_model"],
    )
    @pytest.mark.parametrize("prompt", ["'Why is the Sun yellow?'"])
    def test_python_tool_llm_benchmark_speculative(self, convert_model, convert_draft_model, prompt, sample_args):
        """
        Test Speculative Decoding via GenAI
        """
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model, "--draft_model", convert_draft_model, "-p", prompt] + sample_args
        run_sample(benchmark_py_command)


    @pytest.mark.samples
    @pytest.mark.parametrize("sample_args", 
        [
            ["-d", "cpu", "-n", "1", "--num_steps", "4", "--optimum"],
            ["-d", "cpu", "-n", "1", "--num_steps", "4"],
        ],
    )
    @pytest.mark.parametrize("convert_model", ["tiny-random-latent-consistency"], indirect=True)
    @pytest.mark.parametrize("generate_image_generation_jsonl", [("image_generation.jsonl", image_generation_json)], indirect=True)  
    def test_python_tool_llm_benchmark_jsonl(self, convert_model, generate_image_generation_jsonl, sample_args):
        """
        Test Speculative Decoding via GenAI with JSONL input
        """
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
            "-pf", generate_image_generation_jsonl, 
        ] + sample_args
        run_sample(benchmark_py_command)
        
        
    @pytest.mark.samples
    @pytest.mark.parametrize("sample_args", [["-d", "cpu", "-n", "1", "--num_steps", "4"], ["-d", "cpu", "-n", "1", "--num_steps", "4", "--empty_lora"]])
    @pytest.mark.parametrize("convert_model", ["tiny-random-latent-consistency"], indirect=True)
    @pytest.mark.parametrize("download_model", ["tiny-random-latent-consistency-lora"], indirect=True)
    @pytest.mark.parametrize("generate_image_generation_jsonl", [("image_generation.jsonl", image_generation_json)], indirect=True)
    def test_python_tool_llm_benchmark_jsonl_lora(self, request, convert_model, download_model, generate_image_generation_jsonl, sample_args):
        model_name = request.node.callspec.params['download_model']
        
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
            "-pf", generate_image_generation_jsonl,
            "--lora", f'{download_model}/{model_name}.safetensors',
        ] + sample_args
        run_sample(benchmark_py_command)
        
        
    @pytest.mark.samples
    @pytest.mark.parametrize("sample_args", [["-d", "cpu", "-n", "1", "--num_steps", "4", "--task", "inpainting"]])
    @pytest.mark.parametrize("convert_model", ["tiny-random-latent-consistency"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["overture-creations.png"], indirect=True)
    @pytest.mark.parametrize("download_mask_image", ["overture-creations-mask.png"], indirect=True)
    @pytest.mark.parametrize("generate_image_generation_jsonl", [("image_generation_inpainting.jsonl", image_generation_inpainting_json)], indirect=True)
    def test_python_tool_llm_benchmark_inpainting(self, convert_model, download_test_content, download_mask_image, generate_image_generation_jsonl, sample_args):
        
        # to use the relative media and mask_image paths
        os.chdir(os.path.dirname(download_test_content))

        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
            "-pf", generate_image_generation_jsonl,
        ] + sample_args
        run_sample(benchmark_py_command)


    @pytest.mark.samples
    @pytest.mark.parametrize("sample_args", [["-d", "cpu", "-n", "1", "--num_steps", "4", "--task", "image-to-image"]])
    @pytest.mark.parametrize("convert_model", ["tiny-random-latent-consistency"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["cat.png"], indirect=True)
    @pytest.mark.parametrize("generate_image_generation_jsonl", [("image_generation_i2i.jsonl", image_generation_i2i_json)], indirect=True)
    def test_python_tool_llm_benchmark_i2i(self, convert_model, download_test_content, generate_image_generation_jsonl, sample_args):
        
        # to use the relative media and mask_image paths
        os.chdir(os.path.dirname(download_test_content))

        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
            "-pf", generate_image_generation_jsonl,
        ] + sample_args
        run_sample(benchmark_py_command)
        
        
    @pytest.mark.samples
    @pytest.mark.parametrize("sample_args", [["-d", "cpu", "-n", "1"], ["-d", "cpu", "-n", "1", "--optimum"]])
    @pytest.mark.parametrize("media_file", ["3283_1447_000000.flac"])
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["3283_1447_000.tar.gz"], indirect=True)
    def test_python_tool_llm_benchmark_optimum(self, convert_model, download_test_content, media_file, sample_args):
        media_path = os.path.join(download_test_content, media_file)
        # Run Python benchmark
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
            "--media", media_path,
        ] + sample_args
        run_sample(benchmark_py_command)

    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["bge-small-en-v1.5"], indirect=True)
    @pytest.mark.parametrize("sample_args", [
        ["-d", "cpu", "-n", "2"], 
        ["-d", "cpu", "-n", "2", "--embedding_max_length", "128", "--embedding_normalize", "--embedding_pooling", "mean"], 
        ["-d", "cpu", "-n", "2", "--optimum"], 
        ["-d", "cpu", "-n", "1", "--embedding_max_length", "128", "--embedding_normalize", "--embedding_pooling", "mean", "--optimum"]
    ])
    def test_python_tool_llm_benchmark_text_embeddings(self, convert_model, sample_args):
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'llm_bench/benchmark.py')
        benchmark_py_command = [
            sys.executable, 
            benchmark_script, 
            "-m", convert_model, 
        ] + sample_args
        run_sample(benchmark_py_command)
