type VLMModelType = {
  architecture: string;
  models: Array<{
    name: string;
    loraSupport: boolean;
    links: string[];
    notesLink?: string;
  }>;
};

export const VLM_MODELS: VLMModelType[] = [
  {
    architecture: 'InternVLChat',
    models: [
      {
        name: 'InternVLChatModel',
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
          'https://huggingface.co/OpenGVLab/InternVL3-1B',
          'https://huggingface.co/OpenGVLab/InternVL3-2B',
          'https://huggingface.co/OpenGVLab/InternVL3-8B',
          'https://huggingface.co/OpenGVLab/InternVL3-9B',
          'https://huggingface.co/OpenGVLab/InternVL3-14B'
        ],
        notesLink: '#internvl2-notes',
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
    architecture: 'nanoLLaVA',
    models: [
      {
        name: 'nanoLLaVA',
        loraSupport: false,
        links: ['https://huggingface.co/qnguyen3/nanoLLaVA'],
      },
      {
        name: 'nanoLLaVA-1.5',
        loraSupport: false,
        links: ['https://huggingface.co/qnguyen3/nanoLLaVA-1.5'],
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
    architecture: 'LLaVA-NeXT-Video',
    models: [
      {
        name: 'LLaVA-Next-Video',
        loraSupport: false,
        links: [
          'https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf',
        ],
      },
    ],
  },
  {
    architecture: 'MiniCPMO',
    models: [
      {
        name: 'MiniCPM-o-2_6',
        loraSupport: false,
        links: ['https://huggingface.co/openbmb/MiniCPM-o-2_6'],
        notesLink: '#minicpm-o-notes',
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
        notesLink: '#phi3_v-notes',
      },
    ],
  },
  {
    architecture: 'Phi4MMForCausalLM',
    models: [
      {
        name: 'phi4mm',
        loraSupport: false,
        links: [
          'https://huggingface.co/microsoft/Phi-4-multimodal-instruct',
        ],
        notesLink: '#phi4mm-notes'
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
  {
    architecture: 'Qwen2.5-VL',
    models: [
      {
        name: 'Qwen2.5-VL',
        loraSupport: false,
        links: [
          'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct',
        ],
      },
    ],
  },
  {
    architecture: 'Gemma3ForConditionalGeneration',
    models: [
      {
        name: 'gemma3',
        loraSupport: false,
        links: [
          'https://huggingface.co/google/gemma-3-4b-it',
          'https://huggingface.co/google/gemma-3-12b-it',
          'https://huggingface.co/google/gemma-3-27b-it',
        ],
      },
    ],
  },
];
