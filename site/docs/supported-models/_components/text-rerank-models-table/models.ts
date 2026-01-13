type TextRerankModelType = {
  architecture: string;
  optimumIntelTask: string;
  models: Array<{
    links: string[];
  }>;
};

export const TEXT_RERANK_MODELS: TextRerankModelType[] = [
  {
    architecture: 'BertForSequenceClassification',
    optimumIntelTask: 'text-classification',
    models: [
      {
        links: [
          'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2',
          'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2',
          'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2',
          'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2',
          'https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2',
          'https://huggingface.co/tomaarsen/reranker-MiniLM-L12-gooaq-bce',
        ],
      },
    ],
  },
  {
    architecture: 'XLMRobertaForSequenceClassification',
    optimumIntelTask: 'text-classification',
    models: [
      {
        links: [
          'https://huggingface.co/BAAI/bge-reranker-v2-m3',
        ],
      },
    ],
  },
  {
    architecture: 'ModernBertForSequenceClassification',
    optimumIntelTask: 'text-classification',
    models: [
      {
        links: [
          'https://huggingface.co/tomaarsen/reranker-ModernBERT-base-gooaq-bce',
          'https://huggingface.co/tomaarsen/reranker-ModernBERT-large-gooaq-bce',
          'https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base',
        ],
      },
    ],
  },
  {
    architecture: 'Qwen3ForCausalLM',
    optimumIntelTask: 'text-generation-with-past',
    models: [
      {
        links: [
          'https://huggingface.co/Qwen/Qwen3-Reranker-0.6B',
          'https://huggingface.co/Qwen/Qwen3-Reranker-4B',
          'https://huggingface.co/Qwen/Qwen3-Reranker-8B'
        ],
      },
    ],
  },
];
