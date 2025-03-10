type WhisperModelType = {
  architecture: string;
  models: Array<{
    name: string;
    loraSupport: boolean;
    links: string[];
  }>;
};

export const WHISPER_MODELS: WhisperModelType[] = [
  {
    architecture: 'WhisperForConditionalGeneration',
    models: [
      {
        name: 'Whisper',
        loraSupport: false,
        links: [
          'https://huggingface.co/openai/whisper-tiny',
          'https://huggingface.co/openai/whisper-tiny.en',
          'https://huggingface.co/openai/whisper-base',
          'https://huggingface.co/openai/whisper-base.en',
          'https://huggingface.co/openai/whisper-small',
          'https://huggingface.co/openai/whisper-small.en',
          'https://huggingface.co/openai/whisper-medium',
          'https://huggingface.co/openai/whisper-medium.en',
          'https://huggingface.co/openai/whisper-large-v3',
        ],
      },
      {
        name: 'Distil-Whisper',
        loraSupport: false,
        links: [
          'https://huggingface.co/distil-whisper/distil-small.en',
          'https://huggingface.co/distil-whisper/distil-medium.en',
          'https://huggingface.co/distil-whisper/distil-large-v3',
        ],
      },
    ],
  },
];
