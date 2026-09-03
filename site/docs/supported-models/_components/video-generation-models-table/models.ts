type VideoGenerationModelType = {
  architecture: string;
  textToVideo: boolean;
  imageToVideo: boolean;
  loraSupport: boolean;
  links: string[];
};

export const VIDEO_GENERATION_MODELS: VideoGenerationModelType[] = [
  {
    architecture: 'LTX-Video',
    textToVideo: true,
    imageToVideo: false,
    loraSupport: true,
    links: ['https://huggingface.co/Lightricks/LTX-Video'],
  },
];
