import styles from './styles.module.css';

import Heading from '@theme/Heading';

import Link from '@docusaurus/Link';
import { ImageGeneration } from './components/image-generation';
import { VisualProcessing } from './components/visual-processing';
import { SpeechGeneration } from './components/speech-generation';
import { SpeechRecognition } from './components/speech-recognition';
import { TextGeneration } from './components/text-generation';
import { TextRerank } from './components/text-rerank';
import { MultimodalEmbedding } from './components/multimodal-embedding';
import { VideoGeneration } from './components/video-generation';

export const UseCasesSection = () => (
  <section className={styles.useCasesSection}>
    <Heading as="h2" className={styles.sectionTitle}>
      Use Cases
    </Heading>
    <div className={styles.sectionContent}>
      <TextGeneration />
      <VisualProcessing />
      <ImageGeneration />
      <VideoGeneration />
      <SpeechRecognition />
      <SpeechGeneration />
      <MultimodalEmbedding />
      <TextRerank />
    </div>
    <div className={styles.useCasesFooter}>
      <strong>Looking for more?</strong>&nbsp;See all{' '}
      <Link to="docs/category/use-cases">supported use cases</Link>.
    </div>
  </section>
);
