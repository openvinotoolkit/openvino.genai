type InpaintingModelType = {
  architecture: string;
  loraSupport: boolean;
  links: string[];
};

export const INPAINTING_MODELS: InpaintingModelType[] = [
  {
    architecture: 'Stable Diffusion',
    loraSupport: true,
    links: [
      'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting',
      'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting',
      'https://huggingface.co/botp/stable-diffusion-v1-5-inpainting',
      'https://huggingface.co/parlance/dreamlike-diffusion-1.0-inpainting',
    ],
  },
  {
    architecture: 'Stable Diffusion XL',
    loraSupport: true,
    links: ['https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1'],
  },
];
