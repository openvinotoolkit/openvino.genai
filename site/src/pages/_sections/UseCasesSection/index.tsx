import styles from './styles.module.css';

import Heading from '@theme/Heading';

import { ImageGeneration } from './components/image-generation';
import { ImageProcessing } from './components/image-processing';
import { SpeechRecognition } from './components/speech-recognition';
import { TextGeneration } from './components/text-generation';

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
    </div>
  </section>
);
