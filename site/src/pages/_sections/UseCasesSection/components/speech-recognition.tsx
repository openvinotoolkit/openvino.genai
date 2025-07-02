import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/speech-recognition/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/speech-recognition/_sections/_run_model/_code_example_python.mdx';

export const SpeechRecognition = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Speech Recognition Using Whisper</UseCaseCard.Title>
    <UseCaseCard.Description>
      Convert speech to text using Whisper models for video transcription, meeting notes,
      multilingual audio content processing, and accessibility applications.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Translate foreign language speech directly to English text</li>
      <li>Transcribe audio in multiple languages with automatic language detection</li>
      <li>Generate precise timestamps for synchronized subtitles and captions</li>
      <li>Process long-form audio content (&gt;30 seconds) efficiently</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/speech-recognition" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
