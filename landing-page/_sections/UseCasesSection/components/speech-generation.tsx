import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemJS, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '../../../../docs/use-cases/speech-generation/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '../../../../docs/use-cases/speech-generation/_sections/_run_model/_code_example_python.mdx';
import CodeExampleJS from '../../../../docs/use-cases/speech-generation/_sections/_run_model/_code_example_js.mdx';

export const SpeechGeneration = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Speech Generation Using SpeechT5</UseCaseCard.Title>
    <UseCaseCard.Description>
      Convert text to speech using SpeechT5 TTS models.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Generate natural and expressive speech from text prompts</li>
      <li>Use speaker embeddings for personalized voice synthesis</li>
    </UseCaseCard.Features>
    <UseCaseCard.Code>
      <LanguageTabs>
        <TabItemPython>
          <CodeExamplePython />
        </TabItemPython>
        <TabItemCpp>
          <CodeExampleCpp />
        </TabItemCpp>
        <TabItemJS>
          <CodeExampleJS />
        </TabItemJS>
      </LanguageTabs>
    </UseCaseCard.Code>
    <UseCaseCard.Actions>
      <Button label="Explore Use Case" link="./use-cases/speech-generation" variant="primary" />
      <Button label="View Code Samples" link="./samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
