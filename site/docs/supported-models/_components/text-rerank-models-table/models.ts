type TextRerankModelType = {
  architecture: string;
  models: Array<{
    links: string[];
  }>;
};

export const TEXT_RERANK_MODELS: TextRerankModelType[] = [
  {
    architecture: 'BertForSequenceClassification',
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
];
