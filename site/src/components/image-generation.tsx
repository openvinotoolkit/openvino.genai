import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import { Section } from '@site/src/components/Section';
import { SectionImage } from './Section/section-image';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

import CodeExampleCpp from '@site/docs/use-cases/image-generation/_sections/_run_model/_text2image_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/image-generation/_sections/_run_model/_text2image_python.mdx';

const FEATURES = [
  'Alter parameters (width, height, iterations) and compile model for static size',
  'Load LoRA adapters (in safetensor format) and dynamically switch between them',
  'Generate multiple images per one request',
];

export const ImageGeneration = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Image Generation API</Section.Title>
        <Section.Description>
          A user-friendly image generation API can be used with generative models to improve
          creative tools and increase productivity. For instance, it can be utilized in furniture
          design tools to create various design concepts.
        </Section.Description>
        <SectionImage url={ImagePlaceholder} alt={'Image generation API'} />
      </Section.Column>
      <Section.Column>
        <Section.Features features={FEATURES} />
        <hr />
        <LanguageTabs>
          <TabItemPython>
            <CodeExamplePython />
          </TabItemPython>
          <TabItemCpp>
            <CodeExampleCpp />
          </TabItemCpp>
        </LanguageTabs>
        <hr />
        <ExploreCodeSamples link="docs/category/samples" />
        <GoToDocumentation link="docs/use-cases/image-generation/" />
      </Section.Column>
    </Section.Container>
  );
};
