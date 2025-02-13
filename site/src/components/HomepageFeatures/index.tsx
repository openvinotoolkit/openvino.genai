import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  CodeExample: React.ComponentType<React.ComponentProps<'svg'>>;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Text generation API',
    Svg: require('@site/static/img/text.svg').default,
    CodeExample: require('@site/static/img/text-generation-example.svg').default,
  },
  {
    title: 'Image generation API',
    Svg: require('@site/static/img/image.svg').default,
    CodeExample: require('@site/static/img/image-generation-example.svg').default,
  },
  {
    title: 'Speech to text API',
    Svg: require('@site/static/img/sound-on.svg').default,
    CodeExample: require('@site/static/img/speech-to-text-api-example.svg').default,
  },
];

function Feature({title, Svg, CodeExample}: FeatureItem) {
  return (
    <div>
      <div className={clsx("text--center", styles.featureHeader)}>
        <Svg role="img" />
        <Heading as="h3" className={styles.featureHeading}>{title}</Heading>
      </div>
      <div>
        <CodeExample />
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