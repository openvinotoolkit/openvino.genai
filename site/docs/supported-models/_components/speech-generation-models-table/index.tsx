import React from 'react';
import { BaseModelsTable, LinksCell, StatusCell } from '../base-models-table';
import { SPEECH_GENERATION_MODELS } from './models';

export default function SpeechGenerationModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Models', 'LoRA Support', 'Example HuggingFace Models'];

  const rows = SPEECH_GENERATION_MODELS.map(({ architecture, models }) => (
    <>
      <tr key={architecture}>
        <td rowSpan={models.length}>
          <code>{architecture}</code>
        </td>
        <td>{models[0].name}</td>
        <StatusCell value={models[0].loraSupport} />
        <LinksCell links={models[0].links} />
      </tr>
      {models.slice(1).map(({ name, loraSupport, links }) => (
        <tr key={name}>
          <td>{name}</td>
          <StatusCell value={loraSupport} />
          <LinksCell links={links} />
        </tr>
      ))}
    </>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
