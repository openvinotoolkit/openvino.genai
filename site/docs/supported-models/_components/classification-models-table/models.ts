type ClassificationModelType = {
  architecture: string;
  modalities: string[];
  models: Array<{
    name: string;
    links: string[];
  }>;
};

export const CLASSIFICATION_MODELS: ClassificationModelType[] = [
  {
    architecture: 'Qwen3_5ForSequenceClassification',
    modalities: ['Text', 'Image'],
    models: [
      {
        name: 'MiniCPM-OCR-Jev',
        links: ['https://huggingface.co/openbmb/MiniCPM-OCR-Jev'],
      },
    ],
  },
];
