import Button from '@site/src/components/Button';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import UseCaseCard from './UseCaseCard';

import CodeExampleCpp from '@site/docs/use-cases/text-rerank/_sections/_run_model/_code_example_cpp.mdx';
import CodeExamplePython from '@site/docs/use-cases/text-rerank/_sections/_run_model/_code_example_python.mdx';

export const TextRerank = () => (
  <UseCaseCard>
    <UseCaseCard.Title>Text Rerank for RAG</UseCaseCard.Title>
    <UseCaseCard.Description>
      Boost the relevance and accuracy of your Retrieval-Augmented Generation (RAG) workflows by
      reranking retrieved documents with the TextRerankPipeline.
    </UseCaseCard.Description>
    <UseCaseCard.Features>
      <li>Reorder search results by semantic relevance to the query</li>
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
      <Button label="Explore Use Case" link="docs/use-cases/text-rerank" variant="primary" />
      <Button label="View Code Samples" link="docs/samples" variant="primary" outline />
    </UseCaseCard.Actions>
  </UseCaseCard>
);
