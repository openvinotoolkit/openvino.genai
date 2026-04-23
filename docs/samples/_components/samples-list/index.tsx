import Link from '@docusaurus/Link';
import { usePluginData } from '@docusaurus/useGlobalData';
import Heading from '@theme/Heading';
import React from 'react';

// Shape of data registered by the `genai-samples-docs-plugin` at hub build time.
// Kept in sync with the plugin's GenAISamples type.
type GenAISample = {
  language: string;
  languageTitle: string;
  name: string;
  hasReadme: boolean;
  githubLink: string;
};
type GenAISamples = Record<string, GenAISample[]>;

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
