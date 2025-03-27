import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import { Section } from '@site/src/components/Section';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

import CodeExampleCpp from '@site/docs/use-cases/image-processing/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/image-processing/_sections/_run_model/_code_example_python.mdx';

const FEATURES = [
  'Use different generation parameters (sampling types, etc.)',
  'Optimize for chat scenarios by using chat mode',
  'Pass multiple images to a model',
];

export const ImageProcessing = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Image Processing with Visual Language Models</Section.Title>
        <Section.Description>
          An easy-to-use API for vision language models can power chatbots, AI assistants like
          medical helpers, and AI tools like legal contract creators.
        </Section.Description>
        <Section.Image
          url={ImagePlaceholder}
          alt={'Image processing with Visual Language Models'}
        />
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
        <GoToDocumentation link="docs/use-cases/image-processing" />
      </Section.Column>
    </Section.Container>
  );
};
