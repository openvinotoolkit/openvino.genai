import Link from '@docusaurus/Link';
import { usePluginData } from '@docusaurus/useGlobalData';
import { type GenAISamples } from '@site/src/plugins/genai-samples-docs-plugin';
import Heading from '@theme/Heading';
import React from 'react';

function SamplesListItem({
  item: { language, name, githubLink },
}: {
  item: GenAISamples[string][number];
}): React.JSX.Element {
  return (
    <li>
      <Link href={`./${language}/${name}`}>{name}</Link> (<Link href={githubLink}>GitHub</Link>)
    </li>
  );
}

export default function SamplesList(): React.JSX.Element {
  const samplesMap = usePluginData('genai-samples-docs-plugin') as GenAISamples;

  return (
    <>
      {Object.entries(samplesMap)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([language, samples]) => (
          <div key={language}>
            <Heading as="h2">{samples[0]?.languageTitle}</Heading>
            <ul>
              {samples
                .sort((a, b) => a.name.localeCompare(b.name))
                .map((sample) => (
                  <SamplesListItem key={`${language}-${sample.name}`} item={sample} />
                ))}
            </ul>
          </div>
        ))}
    </>
  );
}
