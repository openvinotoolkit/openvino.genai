import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/image-generation/_sections/_run_model/_text2image_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/image-generation/_sections/_run_model/_text2image_python.mdx';

export const ImageGeneration = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Image Generation Using Diffusers</UseCaseCard.Title>
    <UseCaseCard.Description>
      Create and modify images with diffusion models for art generation, product design, and
      creative applications using Stable Diffusion and similar architectures.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Support for text-to-image, image-to-image, and inpainting pipelines</li>
      <li>Control image generation by adjusting parameters (dimentions, iterations, etc.)</li>
      <li>
        Apply LoRA adapters and dynamically switch between them for artistic styles and
        modifications
      </li>
      <li>Generate multiple images per one request</li>
    </UseCaseCard.Features>
    <UseCaseCard.Code>
      <LanguageTabs>
        <TabItemPython>
          <CodeExamplePython />
        </TabItemPython>
        <TabItemCpp>
          <CodeExampleCpp />
        </TabItemCpp>
      </LanguageTabs>
    </UseCaseCard.Code>
    <UseCaseCard.Actions>
      <Button label="Explore Use Case" link="docs/use-cases/image-generation" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
