import React from 'react';
import { BaseModelsTable, LinksCell, StatusCell } from '../base-models-table';
import { INPAINTING_MODELS } from './models';

export default function InpaintingModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'LoRA Support', 'Example HuggingFace Models'];

  const rows = INPAINTING_MODELS.map(({ architecture, loraSupport, links }) => (
    <tr key={architecture}>
      <td>
        <code style={{ whiteSpace: 'pre' }}>{architecture}</code>
      </td>
      <StatusCell value={loraSupport} />
      <LinksCell links={links} />
    </tr>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
