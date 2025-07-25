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
    parser.add_argument('device')
    args = parser.parse_args()

    device = args.device
    adapter = openvino_genai.Adapter(args.adapter_path)
    adapter_config = openvino_genai.AdapterConfig(adapter)
    pipe = openvino_genai.LLMPipeline(args.models_path, device, adapters=adapter_config)  # register all required adapters here

    print("Generate with LoRA adapter and alpha set to 0.75:")
    print(pipe.generate(args.prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig(adapter, 0.75)))

    print("\n-----------------------------")
    print("Generate without LoRA adapter:")
    print(pipe.generate(args.prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig()))

if '__main__' == __name__:
    main()
