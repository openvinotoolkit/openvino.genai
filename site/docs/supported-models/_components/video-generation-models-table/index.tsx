import React from 'react';
import { BaseModelsTable, LinksCell, StatusCell } from '../base-models-table';
import { VIDEO_GENERATION_MODELS } from './models';

export default function VideoGenerationModelsTable(): React.JSX.Element {
  const headers = [
    'Architecture',
    'Text to Video',
    'Image to Video',
    'Video to Video',
    'LoRA Support',
    'Example HuggingFace Models',
  ];

  const rows = VIDEO_GENERATION_MODELS.map(
    ({ architecture, textToVideo, imageToVideo, videoToVideo, loraSupport, links }) => (
      <tr key={architecture}>
        <td>
          <code style={{ whiteSpace: 'pre' }}>{architecture}</code>
        </td>
        <StatusCell value={textToVideo} />
        <StatusCell value={imageToVideo} />
        <StatusCell value={videoToVideo} />
        <StatusCell value={loraSupport} />
        <LinksCell links={links} />
      </tr>
    )
  );

  return <BaseModelsTable headers={headers} rows={rows} />;
}
