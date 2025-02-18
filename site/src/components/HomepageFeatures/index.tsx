import clsx from 'clsx';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import styles from './styles.module.css';
import React from 'react';


type FeatureItem = {
  title: string;
  Icon: React.ComponentType<React.ComponentProps<'svg'>>;
  code: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Text Generation API',
    Icon: require('@site/static/img/text.svg').default,
    code: `ov_pipe = ov_genai.LLMPipeline("TinyLlama")
print(ov_pipe.generate("The Sun is yellow because"))`,
  },
  {
    title: 'Image Generation API',
    Icon: require('@site/static/img/image.svg').default,
    code: `ov_pipe = ov_genai.Text2ImagePipeline("Flux")
image = ov_pipe.generate("Create beautiful Sun")`,
  },
  {
    title: 'Speech to Text API',
    Icon: require('@site/static/img/sound-on.svg').default,
    code: `ov_pipe = ov_genai.WhisperPipeline("whisper-base")
print(ov_pipe.generate(read_wav("sample.wav)))`,
  },
];

function Feature({title, Icon, code}: FeatureItem) {
  return (
    <div>
      <div className={clsx("text--center", styles.featureHeader)}>
        <Icon role="img" />
        <Heading as="h3" className={styles.featureHeading}>{title}</Heading>
      </div>
      <div>
        <CodeBlock language='python' className={styles.featureCodeExample}>
          {code}
        </CodeBlock>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresList}>
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
