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
];
