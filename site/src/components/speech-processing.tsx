import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import { Section } from '@site/src/components/Section';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

import CodeExampleCpp from '@site/docs/use-cases/speech-processing/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/speech-processing/_sections/_run_model/_code_example_python.mdx';

const FEATURES = [
  'Translate transcription to English',
  'Predict timestamps',
  'Process Long-Form (>30 seconds) audio',
];

export const SpeechProcessing = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Speech Processing API</Section.Title>
        <Section.Description>
          An intuitive speech-to-text API can work with models like Whisper to enable use cases such
          as video transcription, enhancing communication tools.
        </Section.Description>
        <Section.Image url={ImagePlaceholder} alt={'Speech to text'} />
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
        <GoToDocumentation link="docs/use-cases/speech-processing" />
      </Section.Column>
    </Section.Container>
  );
};
