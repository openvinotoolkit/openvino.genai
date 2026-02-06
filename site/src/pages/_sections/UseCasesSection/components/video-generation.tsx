import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/video-generation/_sections/_run_model/_text2video_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/video-generation/_sections/_run_model/_text2video_python.mdx';

export const VideoGeneration = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Video Generation Using Diffusers</UseCaseCard.Title>
    <UseCaseCard.Description>
      Create videos with LTX-Video model using a diffusion-based generation pipeline
      to produce high-quality clips for creative storytelling, marketing content, product demos, and rapid visual prototyping.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Support for text-to-video pipeline</li>
      <li>Control video generation by adjusting parameters (dimensions, iterations, etc.)</li>
      <li>Generate multiple videos per one request</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/video-generation" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
