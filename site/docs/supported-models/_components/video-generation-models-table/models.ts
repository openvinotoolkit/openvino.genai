type VideoGenerationModelType = {
  architecture: string;
  textToVideo: boolean;
  imageToVideo: boolean;
  videoToVideo: boolean;
  loraSupport: boolean;
  links: string[];
};

export const VIDEO_GENERATION_MODELS: VideoGenerationModelType[] = [
  {
    architecture: 'LTX-Video',
    textToVideo: true,
    imageToVideo: true,
    videoToVideo: false,
    loraSupport: true,
    links: ['https://huggingface.co/Lightricks/LTX-Video'],
  },
  {
    architecture: 'LTX-2',
    textToVideo: true,
    imageToVideo: true,
    videoToVideo: true,
    loraSupport: false,
    links: ['https://huggingface.co/Lightricks/LTX-2'],
  },
  {
    architecture: 'LTX-2.3',
    textToVideo: true,
    imageToVideo: true,
    videoToVideo: true,
    loraSupport: false,
    links: ['https://huggingface.co/diffusers/LTX-2.3-Diffusers'],
  },
  {
    architecture: 'LTX-2.5',
    textToVideo: true,
    imageToVideo: true,
    videoToVideo: true,
    loraSupport: false,
    links: ['https://huggingface.co/Lightricks/LTX-2.5-Diffusers'],
  },
];
