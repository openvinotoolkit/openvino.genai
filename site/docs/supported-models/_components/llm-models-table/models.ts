type LLMModelType = {
  architecture: string;
  models: Array<{
    name: string;
    links: string[];
  }>;
};

export const LLM_MODELS: LLMModelType[] = [
  {
    architecture: 'ChatGLMModel',
    models: [
      {
        name: 'ChatGLM',
        links: ['https://huggingface.co/THUDM/chatglm3-6b'],
      },
    ],
  },
  {
    architecture: 'GemmaForCausalLM',
    models: [
      {
        name: 'Gemma',
        links: ['https://huggingface.co/google/gemma-2b-it'],
      },
    ],
  },
  {
    architecture: 'GPTNeoXForCausalLM',
    models: [
      {
        name: 'Dolly',
        links: ['https://huggingface.co/databricks/dolly-v2-3b'],
      },
      {
        name: 'RedPajama',
        links: ['https://huggingface.co/ikala/redpajama-3b-chat'],
      },
    ],
  },
  {
    architecture: 'LlamaForCausalLM',
    models: [
      {
        name: 'Llama 3',
        links: [
          'https://huggingface.co/meta-llama/Meta-Llama-3-8B',
          'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct',
          'https://huggingface.co/meta-llama/Meta-Llama-3-70B',
          'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct',
        ],
      },
      {
        name: 'Llama 2',
        links: [
          'https://huggingface.co/meta-llama/Llama-2-13b-chat-hf',
          'https://huggingface.co/meta-llama/Llama-2-13b-hf',
          'https://huggingface.co/meta-llama/Llama-2-7b-chat-hf',
          'https://huggingface.co/meta-llama/Llama-2-7b-hf',
          'https://huggingface.co/meta-llama/Llama-2-70b-chat-hf',
          'https://huggingface.co/meta-llama/Llama-2-70b-hf',
          'https://huggingface.co/microsoft/Llama2-7b-WhoIsHarryPotter',
        ],
      },
      {
        name: 'OpenLLaMA',
        links: [
          'https://huggingface.co/openlm-research/open_llama_13b',
          'https://huggingface.co/openlm-research/open_llama_3b',
          'https://huggingface.co/openlm-research/open_llama_3b_v2',
          'https://huggingface.co/openlm-research/open_llama_7b',
          'https://huggingface.co/openlm-research/open_llama_7b_v2',
        ],
      },
      {
        name: 'TinyLlama',
        links: ['https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0'],
      },
    ],
  },
  {
    architecture: 'MistralForCausalLM',
    models: [
      {
        name: 'Mistral',
        links: ['https://huggingface.co/mistralai/Mistral-7B-v0.1'],
      },
      {
        name: 'Notus',
        links: ['https://huggingface.co/argilla/notus-7b-v1'],
      },
      {
        name: 'Zephyr',
        links: ['https://huggingface.co/HuggingFaceH4/zephyr-7b-bet1'],
      },
    ],
  },
  {
    architecture: 'PhiForCausalLM',
    models: [
      {
        name: 'Phi',
        links: [
          'https://huggingface.co/microsoft/phi-2',
          'https://huggingface.co/microsoft/phi-1_5',
        ],
      },
    ],
  },
  {
    architecture: 'QWenLMHeadModel',
    models: [
      {
        name: 'Qwen',
        links: [
          'https://huggingface.co/Qwen/Qwen-7B-Chat',
          'https://huggingface.co/Qwen/Qwen-7B-Chat-Int4',
          'https://huggingface.co/Qwen/Qwen1.5-7B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
        ],
      },
    ],
  },
];
