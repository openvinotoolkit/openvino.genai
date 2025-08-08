import React from 'react';
import { TEXT_RERANK_MODELS } from './models';
import { BaseModelsTable, LinksCell } from '../base-models-table';

export default function TextRerankModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Example HuggingFace Models'];

  const rows = TEXT_RERANK_MODELS.map(({ architecture, models }) => (
    <>
      <tr key={architecture}>
        <td rowSpan={models.length}>
          <code>{architecture}</code>
        </td>
        <LinksCell links={models[0].links} />
      </tr>
    </>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
