import React from 'react';
import { BaseModelsTable, LinksCell, StatusCell } from '../base-models-table';
import { TEXT_EMBEDDINGS_MODELS } from './models';

export default function TextEmbeddingsModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'LoRA Support', 'Example HuggingFace Models'];

  const rows = TEXT_EMBEDDINGS_MODELS.map(({ architecture, models }) => (
    <tr key={architecture}>
      <td>
        <code>{architecture}</code>
      </td>
      <StatusCell value={models[0].loraSupport} />
      <LinksCell links={models[0].links} />
    </tr>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
