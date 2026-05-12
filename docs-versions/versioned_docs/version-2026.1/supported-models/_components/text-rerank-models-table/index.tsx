import React from 'react';
import { TEXT_RERANK_MODELS } from './models';
import { BaseModelsTable, LinksCell } from '../base-models-table';

export default function TextRerankModelsTable(): React.JSX.Element {
  const headers = ['Architecture', '`optimum-cli` task', 'Example HuggingFace Models'];

  const rows = TEXT_RERANK_MODELS.map(({ architecture, optimumIntelTask, models }) => (
    <>
      <tr key={architecture}>
        <td rowSpan={models.length}>
          <code>{architecture}</code>
        </td>
        <td rowSpan={models.length}>
          <code>{optimumIntelTask}</code>
        </td>
        <LinksCell links={models[0].links} />
      </tr>
    </>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
