import React from 'react';
import { BaseModelsTable, LinksCell } from '../base-models-table';
import { CLASSIFICATION_MODELS } from './models';

export default function ClassificationModelsTable(): React.JSX.Element {
  const headers = ['Exported Architecture', 'Modalities', 'Supported Model', 'HuggingFace'];

  const rows = CLASSIFICATION_MODELS.map(({ architecture, modalities, models }) => (
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
      <td>{models[0].name}</td>
      <LinksCell links={models[0].links} />
    </tr>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
