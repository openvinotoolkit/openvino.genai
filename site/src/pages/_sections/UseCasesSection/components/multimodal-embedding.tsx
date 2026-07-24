import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/multimodal-embedding/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/multimodal-embedding/_sections/_run_model/_code_example_python.mdx';

export const MultimodalEmbedding = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Semantic Search Using Multimodal Embedding</UseCaseCard.Title>
    <UseCaseCard.Description>
      Generate vector representations for text, images, and videos using a single embedding model.
      Useful for semantic search and retrieval augmented generation (RAG).
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Embed text, images, and videos into a shared space</li>
      <li>Select pooling strategies (CLS, MEAN, LAST_TOKEN)</li>
      <li>Improve retrieval performance with L2 normalization</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/multimodal-embedding" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
