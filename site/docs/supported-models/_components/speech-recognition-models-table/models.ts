type SpeechRecognitionModelType = {
  architecture: string;
  models: Array<{
    name: string;
    links: string[];
  }>;
};

export const SPEECH_RECOGNITION_MODELS: SpeechRecognitionModelType[] = [
  {
    architecture: 'Qwen3ASRForConditionalGeneration',
    models: [
      {
        name: 'Qwen3-ASR',
        links: [
          'https://huggingface.co/Qwen/Qwen3-ASR-0.6B',
          'https://huggingface.co/Qwen/Qwen3-ASR-1.7B',
        ],
      },
    ],
  },
  {
    architecture: 'WhisperForConditionalGeneration',
    models: [
      {
        name: 'Whisper',
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
        links: [
          'https://huggingface.co/distil-whisper/distil-small.en',
          'https://huggingface.co/distil-whisper/distil-medium.en',
          'https://huggingface.co/distil-whisper/distil-large-v3',
        ],
      },
    ],
  },
];
