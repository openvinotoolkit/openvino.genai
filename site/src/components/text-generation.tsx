import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { Section } from './Section';

import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

import CodeExampleCpp from '@site/docs/use-cases/text-generation/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/text-generation/_sections/_run_model/_code_example_python.mdx';

// TODO Consider moving to mdx
const FEATURES = [
  'Use different generation parameters (sampling types, etc.)',
  'Optimize for chat scenarios by using chat mode',
  'Load LoRA adapters and dynamically switch between them without recompilation',
  'Use draft model to accelerate generation via Speculative Decoding',
];

export const TextGeneration = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Text Generation API</Section.Title>
        <Section.Description>
          An easy-to-use API for text generation can work with an LLM model to create chatbots, AI
          assistants like financial helpers, and AI tools like legal contract creators.
        </Section.Description>
        <Section.Image url={ImagePlaceholder} alt={'Text generation API'} />
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
        <ExploreCodeSamples link="docs/samples" />
        <GoToDocumentation link="docs/use-cases/text-generation/" />
      </Section.Column>
    </Section.Container>
  );
};
