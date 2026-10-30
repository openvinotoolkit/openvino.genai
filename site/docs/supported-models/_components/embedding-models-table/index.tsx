import React from 'react';
import { BaseModelsTable, LinksCell } from '../base-models-table';
import { EMBEDDING_MODELS } from './models';

export default function EmbeddingModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Modalities', 'Example HuggingFace Models'];

  const rows = EMBEDDING_MODELS.map(({ architecture, modalities, models }) => (
    <tr key={architecture}>
      <td>
        <code>{architecture}</code>
      </td>
      <td>
        <ul>
          {modalities.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </td>
      <LinksCell links={models[0].links} />
    </tr>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
