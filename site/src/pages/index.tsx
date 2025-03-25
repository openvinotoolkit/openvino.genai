import Heading from '@theme/Heading';
import Layout from '@theme/Layout';

import { ImageGeneration } from '../components/image-generation';
import { ImageProcessing } from '../components/image-processing';
import { Installation } from '../components/Installation/installation';
import { SpeechProcessing } from '../components/speech-processing';
import { TextGeneration } from '../components/text-generation';

import styles from './index.module.css';

import HomepageFeatures from '@site/src/components/HomepageFeatures';

import OpenVinoLogo from '@site/static/img/openvino.svg';
import { JSX } from 'react';

function HomepageHeader() {
  return (
    <header className={styles.banner}>
      <div className="container">
        <Heading as="h1" className={styles.titleContainer}>
          <div className={styles.title}>
            <OpenVinoLogo role="img" />
            <div className={styles.genAITitle}>GenAI</div>
          </div>
        </Heading>
        <p className={styles.subTitle}>Deploy Generative AI with ease</p>
        <p className={styles.description}>
          OpenVINO™ GenAI provides developers the necessary tools to optimize and deploy Generative
          AI models
        </p>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  return (
    <Layout description="Description will go into a meta tag in <head />">
      <main>
        <div className={styles.mainContainer}>
          <HomepageHeader />
          <HomepageFeatures />
        </div>
      </main>
      <div className={styles.contentContainer}>
        <Installation />
        <TextGeneration />
        <ImageGeneration />
        <SpeechProcessing />
        <ImageProcessing />
      </div>
    </Layout>
  );
}
