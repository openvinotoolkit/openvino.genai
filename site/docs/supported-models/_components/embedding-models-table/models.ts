type EmbeddingModelType = {
  architecture: string;
  modalities: string[];
  models: Array<{
    links: string[];
  }>;
};

export const EMBEDDING_MODELS: EmbeddingModelType[] = [
  {
    architecture: 'BertModel',
    modalities: ['Text'],
    models: [
      {
        links: [
          'https://huggingface.co/BAAI/bge-small-en-v1.5',
          'https://huggingface.co/BAAI/bge-base-en-v1.5',
          'https://huggingface.co/BAAI/bge-large-en-v1.5',
          'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2',
          'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1',
          'https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1',
          'https://huggingface.co/WhereIsAI/UAE-Large-V1',
        ],
      },
    ],
  },
  {
    architecture: 'MPNetForMaskedLM',
    modalities: ['Text'],
    models: [
      {
        links: [
          'https://huggingface.co/sentence-transformers/all-mpnet-base-v2',
          'https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1',
        ],
      },
    ],
  },
  {
    architecture: 'RobertaForMaskedLM',
    modalities: ['Text'],
    models: [
      {
        links: ['https://huggingface.co/sentence-transformers/all-distilroberta-v1'],
      },
    ],
  },
  {
    architecture: 'XLMRobertaModel',
    modalities: ['Text'],
    models: [
      {
        links: [
          'https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1',
          'https://huggingface.co/intfloat/multilingual-e5-large-instruct',
          'https://huggingface.co/intfloat/multilingual-e5-large',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen3ForCausalLM',
    modalities: ['Text'],
    models: [
      {
        links: [
          'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B',
          'https://huggingface.co/Qwen/Qwen3-Embedding-4B',
          'https://huggingface.co/Qwen/Qwen3-Embedding-8B',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen3VLModel',
    modalities: ['Text', 'Image', 'Video'],
    models: [
      {
        links: [
          'https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B',
          'https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B',
        ],
      },
    ],
  },
];
