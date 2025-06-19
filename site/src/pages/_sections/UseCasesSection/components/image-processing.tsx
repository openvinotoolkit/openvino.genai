import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/image-processing/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/image-processing/_sections/_run_model/_code_example_python.mdx';

export const ImageProcessing = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Image Processing Using VLMs</UseCaseCard.Title>
    <UseCaseCard.Description>
      Analyze and describe images with Vision Language Models (VLMs) to build AI assistants and
      tools for legal document review, medical analysis, document processing, and visual content
      understanding applications.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Process single or multiple images in a single prompt with detailed text descriptions</li>
      <li>Optimize for conversational scenarios by using chat mode</li>
      <li>Control output with different generation parameters (sampling, temperature, etc.)</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/image-processing" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
