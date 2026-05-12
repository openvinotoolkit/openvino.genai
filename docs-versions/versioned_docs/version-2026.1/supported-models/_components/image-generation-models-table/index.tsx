import React from 'react';
import { BaseModelsTable, LinksCell, StatusCell } from '../base-models-table';
import { IMAGE_GENERATION_MODELS } from './models';

export default function ImageGenerationModelsTable(): React.JSX.Element {
  const headers = [
    'Architecture',
    'Text to Image',
    'Image to Image',
    'Inpainting',
    'LoRA Support',
    'Example HuggingFace Models',
  ];

  const rows = IMAGE_GENERATION_MODELS.map(
    ({ architecture, textToImage, imageToImage, inpainting, loraSupport, links }) => (
      <tr key={architecture}>
        <td>
          <code style={{ whiteSpace: 'pre' }}>{architecture}</code>
        </td>
        <StatusCell value={textToImage} />
        <StatusCell value={imageToImage} />
        <StatusCell value={inpainting} />
        <StatusCell value={loraSupport} />
        <LinksCell links={links} />
      </tr>
    )
  );

  return <BaseModelsTable headers={headers} rows={rows} />;
}
