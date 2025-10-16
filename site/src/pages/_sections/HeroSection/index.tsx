import Heading from '@theme/Heading';

import Button from '@site/src/components/Button';
import OpenVINOLogo from '@site/static/img/openvino.svg';

import PipelinesCarousel from './PipelinesCarousel';
import styles from './styles.module.css';

export const HeroSection = () => (
  <section className={styles.heroSection}>
    <Heading as="h1" className={styles.sectionTitle}>
      <OpenVINOLogo role="img" title="OpenVINO" />
      <span className={styles.genAITitle}>GenAI</span>
    </Heading>
    <div className={styles.sectionContent}>
      <p className={styles.subtitle}>Run Generative AI with ease</p>
      <p className={styles.description}>
        OpenVINO™ GenAI provides optimized pipelines for running generative AI models with maximum
        performance and minimal dependencies
      </p>
      <Button
        label="Get Started"
        link="/docs/getting-started/introduction"
        size="lg"
        variant="secondary"
        className={styles.getStartedButton}
      />
      <PipelinesCarousel className={styles.pipelinesCarousel} />
    </div>
  </section>
);
