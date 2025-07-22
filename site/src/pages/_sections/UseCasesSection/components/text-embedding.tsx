import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/text-embedding/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/text-embedding/_sections/_run_model/_code_example_python.mdx';

export const TextEmbedding = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Text Embedding Pipeline</UseCaseCard.Title>
    <UseCaseCard.Description>
      Generate vector representations for text using embedding models. Useful for semantic search,
      retrieval augmented generation (RAG).
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Supports CLS, MEAN pooling strategies</li>
      <li>Supports embed and query instructions</li>
      <li>Batch embedding for multiple documents</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/text-embedding" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
