import styles from './styles.module.css';

import Heading from '@theme/Heading';

import Link from '@docusaurus/Link';
import { ImageGeneration } from './components/image-generation';
import { ImageProcessing } from './components/image-processing';
import { SpeechRecognition } from './components/speech-recognition';
import { TextGeneration } from './components/text-generation';
import { TextReranking } from './components/text-reranking';

export const UseCasesSection = () => (
  <section className={styles.useCasesSection}>
    <Heading as="h2" className={styles.sectionTitle}>
      Use Cases
    </Heading>
    <div className={styles.sectionContent}>
      <TextGeneration />
      <ImageGeneration />
      <SpeechRecognition />
      <ImageProcessing />
      <TextReranking />
    </div>
    <div className={styles.useCasesFooter}>
      <strong>Looking for more?</strong>&nbsp;See all{' '}
      <Link to="docs/category/use-cases">supported use cases</Link>.
    </div>
  </section>
);
