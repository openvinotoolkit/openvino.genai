type SpeechGenerationModelType = {
  architecture: string;
  models: Array<{
    name: string;
    links: string[];
  }>;
};

export const SPEECH_GENERATION_MODELS: SpeechGenerationModelType[] = [
  {
    architecture: 'SpeechT5ForTextToSpeech',
    models: [
      {
        name: 'SpeechT5 TTS',
        links: ['https://huggingface.co/microsoft/speecht5_tts'],
      },
    ],
  },
  {
    architecture: 'Kokoro',
    models: [
      {
        name: 'Kokoro-82M TTS',
        links: ['https://huggingface.co/hexgrad/Kokoro-82M'],
      },
    ],
  },
  {
    architecture: 'Qwen3TTSForConditionalGeneration',
    models: [
      {
        name: 'Qwen3 TTS Base',
        links: [
          'https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base',
          'https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base',
        ],
      },
      {
        name: 'Qwen3 TTS CustomVoice',
        links: [
          'https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
          'https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        ],
      },
      {
        name: 'Qwen3 TTS VoiceDesign',
        links: ['https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'],
      },
    ],
  },
];
