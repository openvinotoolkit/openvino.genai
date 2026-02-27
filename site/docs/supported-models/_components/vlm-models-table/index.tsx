import Link from '@docusaurus/Link';
import React from 'react';
import { BaseModelsTable, LinksCell } from '../base-models-table';
import { VLM_MODELS } from './models';

export default function VLMModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Models', 'Example HuggingFace Models'];

  const rows = VLM_MODELS.map(({ architecture, models }) => (
    <>
      <tr key={architecture}>
        <td rowSpan={models.length}>
          <code>{architecture}</code>
        </td>
        <td>
          {models[0].name}
          {models[0].notesLink && (
            <>
              &nbsp;(<Link href={models[0].notesLink}>Notes</Link>)
            </>
          )}
        </td>
        <LinksCell links={models[0].links} />
      </tr>
      {models.slice(1).map(({ name, links }) => (
        <tr key={name}>
          <td>{name}</td>
          <LinksCell links={links} />
        </tr>
      ))}
    </>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
