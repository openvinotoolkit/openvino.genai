type VLMModelType = {
  architecture: string;
  models: Array<{
    name: string;
    loraSupport: boolean;
    links: string[];
  }>;
};

export const VLM_MODELS: VLMModelType[] = [
  {
    architecture: 'InternVL2',
    models: [
      {
        name: 'InternVL2',
        loraSupport: false,
        links: [
          'https://huggingface.co/OpenGVLab/InternVL2-1B',
          'https://huggingface.co/OpenGVLab/InternVL2-2B',
          'https://huggingface.co/OpenGVLab/InternVL2-4B',
          'https://huggingface.co/OpenGVLab/InternVL2-8B',
          'https://huggingface.co/OpenGVLab/InternVL2_5-1B',
          'https://huggingface.co/OpenGVLab/InternVL2_5-2B',
          'https://huggingface.co/OpenGVLab/InternVL2_5-4B',
          'https://huggingface.co/OpenGVLab/InternVL2_5-8B',
        ],
      },
    ],
  },
  {
    architecture: 'LLaVA',
    models: [
      {
        name: 'LLaVA-v1.5',
        loraSupport: false,
        links: ['https://huggingface.co/llava-hf/llava-1.5-7b-hf'],
      },
    ],
  },
  {
    architecture: 'LLaVA-NeXT',
    models: [
      {
        name: 'LLaVA-v1.6',
        loraSupport: false,
        links: [
          'https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf',
          'https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf',
          'https://huggingface.co/llava-hf/llama3-llava-next-8b-hf',
        ],
      },
    ],
  },
  {
    architecture: 'MiniCPMV',
    models: [
      {
        name: 'MiniCPM-V-2_6',
        loraSupport: false,
        links: ['https://huggingface.co/openbmb/MiniCPM-V-2_6'],
      },
    ],
  },
  {
    architecture: 'Phi3VForCausalLM',
    models: [
      {
        name: 'phi3_v',
        loraSupport: false,
        links: [
          'https://huggingface.co/microsoft/Phi-3-vision-128k-instruct',
          'https://huggingface.co/microsoft/Phi-3.5-vision-instruct',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen2-VL',
    models: [
      {
        name: 'Qwen2-VL',
        loraSupport: false,
        links: [
          'https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-VL-2B',
          'https://huggingface.co/Qwen/Qwen2-VL-7B',
        ],
      },
    ],
  },
];
