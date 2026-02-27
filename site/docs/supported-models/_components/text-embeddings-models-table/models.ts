type TextEmbeddingsModelType = {
  architecture: string;
  models: Array<{
    links: string[];
  }>;
};

export const TEXT_EMBEDDINGS_MODELS: TextEmbeddingsModelType[] = [
  {
    architecture: 'BertModel',
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
    models: [
      {
        links: ['https://huggingface.co/sentence-transformers/all-distilroberta-v1'],
      },
    ],
  },
  {
    architecture: 'XLMRobertaModel',
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
];
