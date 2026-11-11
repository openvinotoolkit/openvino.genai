import Link from '@docusaurus/Link';
import React from 'react';
import { BaseModelsTable, LinksCell } from '../base-models-table';
import { VLM_MODELS, VLMModel } from './models';

const GENAI_CLI_GUIDE = '/docs/guides/model-preparation/convert-to-openvino#genai-cli';

const ModelNameCell = ({ name, notesLink, genaiCliOnly }: VLMModel) => (
  <td>
    {name}
    {notesLink && (
      <>
        &nbsp;(<Link href={notesLink}>Notes</Link>)
      </>
    )}
    {genaiCliOnly && (
      <>
        &nbsp;(<Link href={GENAI_CLI_GUIDE}>genai-cli only</Link>)
      </>
    )}
  </td>
);

export default function VLMModelsTable(): React.JSX.Element {
  const headers = ['Architecture', 'Models', 'Example HuggingFace Models'];

  const rows = VLM_MODELS.map(({ architecture, models }) => (
    <>
      <tr key={architecture}>
        <td rowSpan={models.length}>
          <code>{architecture}</code>
        </td>
        <ModelNameCell {...models[0]} />
        <LinksCell links={models[0].links} />
      </tr>
      {models.slice(1).map((model) => (
        <tr key={model.name}>
          <ModelNameCell {...model} />
          <LinksCell links={model.links} />
        </tr>
      ))}
    </>
  ));

  return <BaseModelsTable headers={headers} rows={rows} />;
}
