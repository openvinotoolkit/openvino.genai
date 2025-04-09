import { Plugin } from '@docusaurus/types';
import fs from 'fs';
import path from 'path';

type GenAISample = {
  language: string;
  languageTitle: string;
  name: string;
  hasReadme: boolean;
  githubLink: string;
};

export type GenAISamples = {
  [language: string]: GenAISample[];
};

const LANGUAGE_TITLES = {
  c: 'C',
  cpp: 'C++',
  js: 'JavaScript',
  python: 'Python',
} as const;

async function findSamples(): Promise<GenAISamples> {
  const samplesMap: GenAISamples = {};
  const samplesPath = path.join('..', 'samples'); // Relative to docusaurus.config.ts

  for await (const dir of await fs.promises.opendir(samplesPath)) {
    const dirPath = path.join(samplesPath, dir.name);
    if (dir.isDirectory()) {
      const language = dir.name;
      for await (const subdir of await fs.promises.opendir(dirPath)) {
        if (subdir.isDirectory()) {
          if (!samplesMap[language]) {
            samplesMap[language] = [];
          }
          const hasReadme = await fs.promises
            .access(path.join(dirPath, subdir.name, 'README.md'))
            .then(() => true)
            .catch(() => false);

          const githubLink = `https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/${language}/${subdir.name}`;

          samplesMap[language].push({
            language,
            languageTitle: LANGUAGE_TITLES[language] || language,
            name: subdir.name,
            hasReadme,
            githubLink,
          });
        }
      }
    }
  }
  return samplesMap;
}

async function generateSamplesDocs(samplesMap: GenAISamples): Promise<void> {
  const docsSamplesPath = path.join('docs', 'samples');
  for (const [language, samples] of Object.entries(samplesMap)) {
    const languageDirPath = path.join(docsSamplesPath, language);
    const languageTitle = samples[0]?.languageTitle;
    await fs.promises.mkdir(languageDirPath, { recursive: true });
    await generateCategory(languageTitle, languageDirPath);
    for (const sample of samples) {
      await generateSampleDocFile(sample, languageDirPath);
    }
  }
}

async function generateCategory(language: string, dirPath: string): Promise<void> {
  const content = {
    label: language,
    link: {
      type: 'generated-index',
      description: `OpenVINO GenAI ${language} samples`,
    },
  };

  await fs.promises.writeFile(
    path.join(dirPath, '_category_.json'),
    JSON.stringify(content, null, 2)
  );
}

async function generateSampleDocFile(sample: GenAISample, dirPath: string): Promise<void> {
  const sampleDocPath = path.join(dirPath, `${sample.name}.mdx`);

  const readmeImportContent = `
import SampleReadme from '@site/../samples/${sample.language}/${sample.name}/README.md';

<Button label="View on GitHub" variant="primary" size="sm" outline link="${sample.githubLink}" />

<SampleReadme />`;

  const fallbackContent = `
# OpenVINO GenAI ${sample.languageTitle} Samples

Refer to the [${sample.languageTitle} ${sample.name} sample](${sample.githubLink}) in GitHub for more information about OpenVINO GenAI ${sample.languageTitle} API.
`;

  const content = `---
hide_title: true
sidebar_label: ${sample.name}
---
${sample.hasReadme ? readmeImportContent : fallbackContent}`;

  await fs.promises.writeFile(sampleDocPath, content);
}

export default async function GenAISamplesDocsPlugin(): Promise<Plugin> {
  return {
    name: 'genai-samples-docs-plugin',
    async loadContent() {
      return findSamples();
    },
    async contentLoaded({ content, actions }) {
      actions.setGlobalData(content);
    },
    async extendCli(cli) {
      cli
        .command('generate-samples-docs')
        .description('Generate documentation pages for samples from `{repositoryRoot}/samples`')
        .action(async () => {
          const samplesMap = await findSamples();
          await generateSamplesDocs(samplesMap);
        });
    },
  };
}
