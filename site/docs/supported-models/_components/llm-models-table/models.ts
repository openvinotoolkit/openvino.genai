type LLMModelType = {
  architecture: string;
  models: Array<{
    name: string;
    links: string[];
  }>;
};

export const LLM_MODELS: LLMModelType[] = [
  {
    architecture: 'AquilaModel',
    models: [
      {
        name: 'Aquila',
        links: [
          'https://huggingface.co/BAAI/Aquila-7B',
          'https://huggingface.co/BAAI/AquilaChat-7B',
          'https://huggingface.co/BAAI/Aquila2-7B',
          'https://huggingface.co/BAAI/AquilaChat2-7B',
        ],
      },
    ],
  },
  {
    architecture: 'ArcticForCausalLM',
    models: [
      {
        name: 'Snowflake',
        links: [
          'https://huggingface.co/Snowflake/snowflake-arctic-instruct',
          'https://huggingface.co/Snowflake/snowflake-arctic-base',
        ],
      },
    ],
  },
  {
    architecture: 'BaichuanForCausalLM',
    models: [
      {
        name: 'Baichuan2',
        links: [
          'https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat',
          'https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat',
        ],
      },
    ],
  },
  {
    architecture: 'BloomForCausalLM',
    models: [
      {
        name: 'Bloom',
        links: [
          'https://huggingface.co/bigscience/bloom-560m',
          'https://huggingface.co/bigscience/bloom-1b1',
          'https://huggingface.co/bigscience/bloom-1b7',
          'https://huggingface.co/bigscience/bloom-3b',
          'https://huggingface.co/bigscience/bloom-7b1',
        ],
      },
      {
        name: 'Bloomz',
        links: [
          'https://huggingface.co/bigscience/bloomz-560m',
          'https://huggingface.co/bigscience/bloomz-1b1',
          'https://huggingface.co/bigscience/bloomz-1b7',
          'https://huggingface.co/bigscience/bloomz-3b',
          'https://huggingface.co/bigscience/bloomz-7b1',
        ],
      },
    ],
  },
  {
    architecture: 'ChatGLMModel',
    models: [
      {
        name: 'ChatGLM',
        links: [
          'https://huggingface.co/THUDM/chatglm2-6b',
          'https://huggingface.co/THUDM/chatglm3-6b',
          'https://huggingface.co/THUDM/glm-4-9b',
          'https://huggingface.co/THUDM/glm-4-9b-chat',
        ],
      },
    ],
  },
  {
    architecture: 'CodeGenForCausalLM',
    models: [
      {
        name: 'CodeGen',
        links: [
          'https://huggingface.co/Salesforce/codegen-350m-multi',
          'https://huggingface.co/Salesforce/codegen-2B-multi',
          'https://huggingface.co/Salesforce/codegen-6B-multi',
          'https://huggingface.co/Salesforce/codegen-16B-multi',
          'https://huggingface.co/Salesforce/codegen-350m-mono',
          'https://huggingface.co/Salesforce/codegen-2B-mono',
          'https://huggingface.co/Salesforce/codegen-6B-mono',
          'https://huggingface.co/Salesforce/codegen-16B-mono',
          'https://huggingface.co/Salesforce/codegen2-1B_P',
          'https://huggingface.co/Salesforce/codegen2-3_7B_P',
          'https://huggingface.co/Salesforce/codegen2-7B_P',
          'https://huggingface.co/Salesforce/codegen2-16B_P',
        ],
      },
    ],
  },
  {
    architecture: 'CohereForCausalLM',
    models: [
      {
        name: 'Aya',
        links: [
          'https://huggingface.co/CohereLabs/aya-23-8B',
          'https://huggingface.co/CohereLabs/aya-expanse-8b',
          'https://huggingface.co/CohereLabs/aya-23-35B',
        ],
      },
      {
        name: 'C4AI Command R',
        links: [
          'https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024',
          'https://huggingface.co/CohereLabs/c4ai-command-r-v01',
        ],
      },
    ],
  },
  {
    architecture: 'DbrxForCausalLM',
    models: [
      {
        name: 'DBRX',
        links: [
          'https://huggingface.co/databricks/dbrx-instruct',
          'https://huggingface.co/databricks/dbrx-base',
        ],
      },
    ],
  },
  {
    architecture: 'DeciLMForCausalLM',
    models: [
      {
        name: 'DeciLM',
        links: [
          'https://huggingface.co/Deci/DeciLM-7B',
          'https://huggingface.co/Deci/DeciLM-7B-instruct',
        ],
      },
    ],
  },
  {
    architecture: 'DeepseekForCausalLM',
    models: [
      {
        name: 'DeepSeek-MoE',
        links: [
          'https://huggingface.co/deepseek-ai/deepseek-moe-16b-base',
          'https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat',
        ],
      },
    ],
  },
  {
    architecture: 'DeepseekV2ForCausalLM',
    models: [
      {
        name: 'DeepSeekV2',
        links: [
          'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite',
          'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat',
          'https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        ],
      },
    ],
  },
  {
    architecture: 'DeepseekV3ForCausalLM',
    models: [
      {
        name: 'DeepSeekV3',
        links: [
          'https://huggingface.co/deepseek-ai/DeepSeek-V3',
          'https://huggingface.co/deepseek-ai/DeepSeek-V3-Base',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero',
        ],
      },
    ],
  },
  {
    architecture: 'ExaoneForCausalLM',
    models: [
      {
        name: 'Exaone',
        links: [
          'https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
          'https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
          'https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct',
          'https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',
        ],
      },
    ],
  },
  {
    architecture: 'FalconForCausalLM',
    models: [
      {
        name: 'Falcon',
        links: [
          'https://huggingface.co/tiiuae/falcon-11B',
          'https://huggingface.co/tiiuae/falcon-7b',
          'https://huggingface.co/tiiuae/falcon-7b-instruct',
          'https://huggingface.co/tiiuae/falcon-40b',
          'https://huggingface.co/tiiuae/falcon-40b-instruct',
        ],
      },
    ],
  },
  {
    architecture: 'GemmaForCausalLM',
    models: [
      {
        name: 'Gemma',
        links: [
          'https://huggingface.co/google/gemma-2b',
          'https://huggingface.co/google/gemma-2b-it',
          'https://huggingface.co/google/gemma-1.1-2b-it',
          'https://huggingface.co/google/codegemma-2b',
          'https://huggingface.co/google/codegemma-1.1-2b',
          'https://huggingface.co/google/gemma-7b',
          'https://huggingface.co/google/gemma-7b-it',
          'https://huggingface.co/google/gemma-1.1-7b-it',
          'https://huggingface.co/google/codegemma-7b',
          'https://huggingface.co/google/codegemma-7b-it',
          'https://huggingface.co/google/codegemma-1.1-7b-it',
        ],
      },
    ],
  },
  {
    architecture: 'Gemma2ForCausalLM',
    models: [
      {
        name: 'Gemma2',
        links: [
          'https://huggingface.co/google/gemma-2-2b',
          'https://huggingface.co/google/gemma-2-2b-it',
          'https://huggingface.co/google/gemma-2-9b',
          'https://huggingface.co/google/gemma-2-9b-it',
          'https://huggingface.co/google/gemma-2-27b',
          'https://huggingface.co/google/gemma-2-27b-it',
        ],
      },
    ],
  },
  {
    architecture: 'Gemma3ForCausalLM',
    models: [
      {
        name: 'Gemma3',
        links: [
          'https://huggingface.co/google/gemma-3-1b-it',
          'https://huggingface.co/google/gemma-3-1b-pt',
        ],
      },
    ],
  },
  {
    architecture: 'GlmForCausalLM',
    models: [
      {
        name: 'GLM',
        links: [
          'https://huggingface.co/THUDM/glm-edge-1.5b-chat',
          'https://huggingface.co/THUDM/glm-edge-4b-chat',
          'https://huggingface.co/THUDM/glm-4-9b-hf',
          'https://huggingface.co/THUDM/glm-4-9b-chat-hf',
          'https://huggingface.co/THUDM/glm-4-9b-chat-1m-hf',
        ],
      },
    ],
  },
  {
    architecture: 'GPT2LMHeadModel',
    models: [
      {
        name: 'GPT2',
        links: [
          'https://huggingface.co/openai-community/gpt2',
          'https://huggingface.co/openai-community/gpt2-medium',
          'https://huggingface.co/openai-community/gpt2-large',
          'https://huggingface.co/openai-community/gpt2-xl',
          'https://huggingface.co/distilbert/distilgpt2',
        ],
      },
      {
        name: 'CodeParrot',
        links: [
          'https://huggingface.co/codeparrot/codeparrot-small',
          'https://huggingface.co/codeparrot/codeparrot-small-code-to-text',
          'https://huggingface.co/codeparrot/codeparrot-small-text-to-code',
          'https://huggingface.co/codeparrot/codeparrot-small-multi',
          'https://huggingface.co/codeparrot/codeparrot',
        ],
      },
    ],
  },
  {
    architecture: 'GPTBigCodeForCausalLM',
    models: [
      {
        name: 'StarCoder',
        links: [
          'https://huggingface.co/bigcode/starcoderbase-1b',
          'https://huggingface.co/bigcode/starcoderbase-3b',
          'https://huggingface.co/bigcode/starcoderbase-7b',
          'https://huggingface.co/bigcode/starcoderbase',
          'https://huggingface.co/bigcode/starcoder',
          'https://huggingface.co/bigcode/octocoder',
          'https://huggingface.co/HuggingFaceH4/starchat-alpha',
          'https://huggingface.co/HuggingFaceH4/starchat-beta',
        ],
      },
    ],
  },
  {
    architecture: 'GPTJForCausalLM',
    models: [
      {
        name: 'GPT-J',
        links: [
          'https://huggingface.co/EleutherAI/gpt-j-6b',
          'https://huggingface.co/crumb/Instruct-GPT-J',
        ],
      },
    ],
  },
  {
    architecture: 'GPTNeoForCausalLM',
    models: [
      {
        name: 'GPT Neo',
        links: [
          'https://huggingface.co/EleutherAI/gpt-neo-1.3B',
          'https://huggingface.co/EleutherAI/gpt-neo-2.7B',
        ],
      },
    ],
  },
  {
    architecture: 'GPTNeoXForCausalLM',
    models: [
      {
        name: 'GPT NeoX',
        links: ['https://huggingface.co/EleutherAI/gpt-neox-20b'],
      },
      {
        name: 'Dolly',
        links: [
          'https://huggingface.co/databricks/dolly-v2-3b',
          'https://huggingface.co/databricks/dolly-v2-7b',
          'https://huggingface.co/databricks/dolly-v2-12b',
        ],
      },
      {
        name: 'RedPajama',
        links: [
          'https://huggingface.co/ikala/redpajama-3b-chat',
          'https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1',
          'https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1',
          'https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat',
          'https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct',
        ],
      },
    ],
  },
  {
    architecture: 'GPTNeoXJapaneseForCausalLM',
    models: [
      {
        name: 'GPT NeoX Japanese',
        links: ['https://huggingface.co/abeja/gpt-neox-japanese-2.7b'],
      },
    ],
  },
  {
    architecture: 'GptOssForCausalLM',
    models: [
      {
        name: 'GPT-OSS',
        links: [
          'https://huggingface.co/openai/gpt-oss-20b',
        ],
      },
    ],
  },
  {
    architecture: 'GraniteForCausalLM',
    models: [
      {
        name: 'Granite',
        links: [
          'https://huggingface.co/ibm-granite/granite-3.2-2b-instruct',
          'https://huggingface.co/ibm-granite/granite-3.2-8b-instruct',
          'https://huggingface.co/ibm-granite/granite-3.1-2b-instruct',
          'https://huggingface.co/ibm-granite/granite-3.1-8b-instruct',
          'https://huggingface.co/ibm-granite/granite-3.0-2b-instruct',
          'https://huggingface.co/ibm-granite/granite-3.0-8b-instruct',
        ],
      },
    ],
  },
  {
    architecture: 'GraniteMoeForCausalLM',
    models: [
      {
        name: 'GraniteMoE',
        links: [
          'https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct',
          'https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct',
          'https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct',
          'https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-instruct',
        ],
      },
    ],
  },
  {
    architecture: 'InternLMForCausalLM',
    models: [
      {
        name: 'InternLM',
        links: [
          'https://huggingface.co/internlm/internlm-chat-7b',
          'https://huggingface.co/internlm/internlm-7b',
        ],
      },
    ],
  },
  {
    architecture: 'InternLM2ForCausalLM',
    models: [
      {
        name: 'InternLM2',
        links: [
          'https://huggingface.co/internlm/internlm2-chat-1_8b',
          'https://huggingface.co/internlm/internlm2-1_8b',
          'https://huggingface.co/internlm/internlm2-chat-7b',
          'https://huggingface.co/internlm/internlm2-7b',
          'https://huggingface.co/internlm/internlm2-chat-20b',
          'https://huggingface.co/internlm/internlm2-20b',
          'https://huggingface.co/internlm/internlm2_5-1_8b-chat',
          'https://huggingface.co/internlm/internlm2_5-1_8b',
          'https://huggingface.co/internlm/internlm2_5-7b-chat',
          'https://huggingface.co/internlm/internlm2_5-7b',
          'https://huggingface.co/internlm/internlm2_5-20b-chat',
          'https://huggingface.co/internlm/internlm2_5-20b',
        ],
      },
    ],
  },
  {
    architecture: 'JAISLMHeadModel',
    models: [
      {
        name: 'Jais',
        links: [
          'https://huggingface.co/inceptionai/jais-13b-chat',
          'https://huggingface.co/inceptionai/jais-13b',
        ],
      },
    ],
  },
  {
    architecture: 'LlamaForCausalLM',
    models: [
      {
        name: 'Llama 3',
        links: [
          'https://huggingface.co/meta-llama/Llama-3.2-1B',
          'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct',
          'https://huggingface.co/meta-llama/Llama-3.2-3B',
          'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct',
          'https://huggingface.co/meta-llama/Llama-3.1-8B',
          'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct',
          'https://huggingface.co/meta-llama/Meta-Llama-3-8B',
          'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct',
          'https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct',
          'https://huggingface.co/meta-llama/Llama-3.1-70B',
          'https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct',
          'https://huggingface.co/meta-llama/Meta-Llama-3-70B',
          'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
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
        name: 'Falcon3',
        links: [
          'https://huggingface.co/tiiuae/Falcon3-1B-Instruct',
          'https://huggingface.co/tiiuae/Falcon3-1B-Base',
          'https://huggingface.co/tiiuae/Falcon3-3B-Instruct',
          'https://huggingface.co/tiiuae/Falcon3-3B-Base',
          'https://huggingface.co/tiiuae/Falcon3-7B-Instruct',
          'https://huggingface.co/tiiuae/Falcon3-7B-Base',
          'https://huggingface.co/tiiuae/Falcon3-10B-Instruct',
          'https://huggingface.co/tiiuae/Falcon3-10B-Base',
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
    architecture: 'MPTForCausalLM',
    models: [
      {
        name: 'MPT',
        links: [
          'https://huggingface.co/mosaicml/mpt-7b',
          'https://huggingface.co/mosaicml/mpt-7b-instruct',
          'https://huggingface.co/mosaicml/mpt-7b-chat',
          'https://huggingface.co/mosaicml/mpt-30b',
          'https://huggingface.co/mosaicml/mpt-30b-instruct',
          'https://huggingface.co/mosaicml/mpt-30b-chat',
        ],
      },
    ],
  },
  {
    architecture: 'MiniCPMForCausalLM',
    models: [
      {
        name: 'MiniCPM',
        links: [
          'https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16',
          'https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp16',
          'https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32',
          'https://huggingface.co/openbmb/MiniCPM-2B-dpo-fp32',
          'https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16',
          'https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16',
          'https://huggingface.co/openbmb/MiniCPM4-0.5B',
          'https://huggingface.co/openbmb/MiniCPM4-8B',
        ],
      },
    ],
  },
  {
    architecture: 'MiniCPM3ForCausalLM',
    models: [
      {
        name: 'MiniCPM3',
        links: ['https://huggingface.co/openbmb/MiniCPM3-4B'],
      },
    ],
  },
  {
    architecture: 'MistralForCausalLM',
    models: [
      {
        name: 'Mistral',
        links: [
          'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1',
          'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2',
          'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3',
          'https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407',
          'https://huggingface.co/mistralai/Mistral-Nemo-Base-2407',
          'https://huggingface.co/mistralai/Mistral-7B-v0.1',
          'https://huggingface.co/mistralai/Mistral-7B-v0.3',
        ],
      },
      {
        name: 'Notus',
        links: ['https://huggingface.co/argilla/notus-7b-v1'],
      },
      {
        name: 'Zephyr',
        links: ['https://huggingface.co/HuggingFaceH4/zephyr-7b-beta'],
      },
      {
        name: 'Neural Chat',
        links: [
          'https://huggingface.co/Intel/neural-chat-7b-v3-3',
          'https://huggingface.co/Intel/neural-chat-7b-v3-2',
          'https://huggingface.co/Intel/neural-chat-7b-v3-1',
          'https://huggingface.co/Intel/neural-chat-7b-v3',
        ],
      },
    ],
  },
  {
    architecture: 'MixtralForCausalLM',
    models: [
      {
        name: 'Mixtral',
        links: [
          'https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1',
          'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1',
        ],
      },
    ],
  },
  {
    architecture: 'OlmoForCausalLM',
    models: [
      {
        name: 'OLMo',
        links: [
          'https://huggingface.co/allenai/OLMo-1B-hf',
          'https://huggingface.co/allenai/OLMo-7B-hf',
          'https://huggingface.co/allenai/OLMo-7B-Twin-2T-hf',
          'https://huggingface.co/allenai/OLMo-7B-Instruct-hf',
          'https://huggingface.co/allenai/OLMo-7B-0724-Instruct-hf',
          'https://huggingface.co/allenai/OLMo-7B-0724-SFT-hf',
        ],
      },
    ],
  },
  {
    architecture: 'OPTForCausalLM',
    models: [
      {
        name: 'OPT',
        links: [
          'https://huggingface.co/facebook/opt-125m',
          'https://huggingface.co/facebook/opt-350m',
          'https://huggingface.co/facebook/opt-1.3b',
          'https://huggingface.co/facebook/opt-2.7b',
          'https://huggingface.co/facebook/opt-6.7b',
          'https://huggingface.co/facebook/opt-13b',
        ],
      },
    ],
  },
  {
    architecture: 'OrionForCausalLM',
    models: [
      {
        name: 'Orion',
        links: [
          'https://huggingface.co/OrionStarAI/Orion-14B-Chat',
          'https://huggingface.co/OrionStarAI/Orion-14B-LongChat',
          'https://huggingface.co/OrionStarAI/Orion-14B-Base',
        ],
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
    architecture: 'Phi3ForCausalLM',
    models: [
      {
        name: 'Phi3',
        links: [
          'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct',
          'https://huggingface.co/microsoft/Phi-3-mini-128k-instruct',
          'https://huggingface.co/microsoft/Phi-3-medium-4k-instruct',
          'https://huggingface.co/microsoft/Phi-3-medium-128k-instruct',
          'https://huggingface.co/microsoft/Phi-3.5-mini-instruct',
          'https://huggingface.co/microsoft/Phi-4-mini-instruct',
          'https://huggingface.co/microsoft/phi-4',
          'https://huggingface.co/microsoft/Phi-4-reasoning',
        ],
      },
    ],
  },
  {
    architecture: 'PhimoeForCausalLM',
    models: [
      {
        name: 'Phi-3.5-MoE',
        links: ['https://huggingface.co/microsoft/Phi-3.5-MoE-instruct'],
      },
    ],
  },
  {
    architecture: 'QWenLMHeadModel',
    models: [
      {
        name: 'Qwen',
        links: [
          'https://huggingface.co/Qwen/Qwen-1_8B-Chat',
          'https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4',
          'https://huggingface.co/Qwen/Qwen-1_8B',
          'https://huggingface.co/Qwen/Qwen-7B-Chat',
          'https://huggingface.co/Qwen/Qwen-7B-Chat-Int4',
          'https://huggingface.co/Qwen/Qwen-7B',
          'https://huggingface.co/Qwen/Qwen-14B-Chat',
          'https://huggingface.co/Qwen/Qwen-14B-Chat-Int4',
          'https://huggingface.co/Qwen/Qwen-14B',
          'https://huggingface.co/Qwen/Qwen-72B-Chat',
          'https://huggingface.co/Qwen/Qwen-72B-Chat-Int4',
          'https://huggingface.co/Qwen/Qwen-72B',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen2ForCausalLM',
    models: [
      {
        name: 'Qwen2',
        links: [
          'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct',
          'https://huggingface.co/Qwen/Qwen2.5-72B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-0.5B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-1.5B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-7B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-72B-Instruct',
          'https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-4B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-7B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-14B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-32B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
          'https://huggingface.co/Qwen/QwQ-32B',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
          'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen2MoeForCausalLM',
    models: [
      {
        name: 'Qwen2MoE',
        links: [
          'https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct',
          'https://huggingface.co/Qwen/Qwen2-57B-A14B',
          'https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat',
          'https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen3ForCausalLM',
    models: [
      {
        name: 'Qwen3',
        links: [
          'https://huggingface.co/Qwen/Qwen3-0.6B',
          'https://huggingface.co/Qwen/Qwen3-1.7B',
          'https://huggingface.co/Qwen/Qwen3-4B',
          'https://huggingface.co/Qwen/Qwen3-8B',
          'https://huggingface.co/Qwen/Qwen3-14B',
          'https://huggingface.co/Qwen/Qwen3-32B',
          'https://huggingface.co/Qwen/Qwen3-0.6B-Base',
          'https://huggingface.co/Qwen/Qwen3-1.7B-Base',
          'https://huggingface.co/Qwen/Qwen3-4B-Base',
          'https://huggingface.co/Qwen/Qwen3-8B-Base',
          'https://huggingface.co/Qwen/Qwen3-14B-Base',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen3MoeForCausalLM',
    models: [
      {
        name: 'Qwen3MoE',
        links: [
          'https://huggingface.co/Qwen/Qwen3-30B-A3B',
          'https://huggingface.co/Qwen/Qwen3-30B-A3B-Base',
        ],
      },
    ],
  },
  {
    architecture: 'StableLmForCausalLM',
    models: [
      {
        name: 'StableLM',
        links: [
          'https://huggingface.co/stabilityai/stablelm-zephyr-3b',
          'https://huggingface.co/stabilityai/stablelm-2-1_6b',
          'https://huggingface.co/stabilityai/stablelm-2-12b',
          'https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b',
          'https://huggingface.co/stabilityai/stablelm-3b-4e1t',
        ],
      },
    ],
  },
  {
    architecture: 'Starcoder2ForCausalLM',
    models: [
      {
        name: 'Startcoder2',
        links: [
          'https://huggingface.co/bigcode/starcoder2-3b',
          'https://huggingface.co/bigcode/starcoder2-7b',
          'https://huggingface.co/bigcode/starcoder2-15b',
        ],
      },
    ],
  },
  {
    architecture: 'XGLMForCausalLM',
    models: [
      {
        name: 'XGLM',
        links: [
          'https://huggingface.co/facebook/xglm-564M',
          'https://huggingface.co/facebook/xglm-1.7B',
          'https://huggingface.co/facebook/xglm-2.9B',
          'https://huggingface.co/facebook/xglm-4.5B',
          'https://huggingface.co/facebook/xglm-7.5B',
        ],
      },
    ],
  },
  {
    architecture: 'XverseForCausalLM',
    models: [
      {
        name: 'Xverse',
        links: [
          'https://huggingface.co/xverse/XVERSE-7B',
          'https://huggingface.co/xverse/XVERSE-7B-Chat',
          'https://huggingface.co/xverse/XVERSE-13B',
          'https://huggingface.co/xverse/XVERSE-13B-Chat',
          'https://huggingface.co/xverse/XVERSE-65B',
          'https://huggingface.co/xverse/XVERSE-65B-Chat',
        ],
      },
    ],
  },
];
