#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path')
    parser.add_argument('adapter_path')
    parser.add_argument('prompt')
    args = parser.parse_args()


    device = 'CPU'  # GPU can be used as well
    adapter = openvino_genai.Adapter(args.adapter_path)
    adapter_config = openvino_genai.AdapterConfig()
    adapter_config.add(adapter, 0.75)

    pipe = openvino_genai.LLMPipeline(args.model_dir, device, adapters=adapter_config)

    print("Generate with LoRA adapter and alpha set to 0.75:")
    print(pipe.generate(args.prompt, max_new_tokens=100, adapters=adapter_config))

    print("-----------------------------")
    print("Generate without LoRA adapter:")
    print(pipe.generate(args.prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig()))

if '__main__' == __name__:
    main()
