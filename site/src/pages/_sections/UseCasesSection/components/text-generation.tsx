import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';

import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/text-generation/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/text-generation/_sections/_run_model/_code_example_python.mdx';

export const TextGeneration = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Text Generation Using LLMs</UseCaseCard.Title>
    <UseCaseCard.Description>
      Create chatbots, text summarization, content generation, and question-answering applications
      with state-of-the-art Large Language Models (LLMs).
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Control output with different generation parameters (sampling, temperature, etc.)</li>
      <li>Optimize for conversational scenarios by using chat mode</li>
      <li>Apply LoRA adapters and dynamically switch between them without recompilation</li>
      <li>Accelerate generation using draft models via Speculative Decoding</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/text-generation" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
