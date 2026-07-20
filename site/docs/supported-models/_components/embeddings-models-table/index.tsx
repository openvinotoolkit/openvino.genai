import React from 'react';
import { BaseModelsTable, LinksCell } from '../base-models-table';
import { EMBEDDINGS_MODELS } from './models';

export default function EmbeddingsModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Modality', 'Example HuggingFace Models'];

  const rows = EMBEDDINGS_MODELS.map(({ architecture, modality, models }) => (
    <tr key={architecture}>
      <td>
        <code>{architecture}</code>
      </td>
      <td>{modality}</td>
      <LinksCell links={models[0].links} />
    </tr>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
