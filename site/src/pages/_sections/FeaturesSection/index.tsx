import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';

import styles from './styles.module.css';

import { FeatureItem } from './FeatureItem';

export const FeaturesSection = () => (
  <section className={styles.featuresSection}>
    <Heading as="h2" className={styles.sectionTitle}>
      Features and Benefits
    </Heading>
    <div className={styles.sectionContent}>
      <FeatureItem icon="🚀" title="Optimized Performance">
        Built for speed with hardware-specific optimizations for Intel CPUs, GPUs, and NPUs.
        Advanced techniques like speculative decoding, KV-cache optimization, and other deliver
        maximum inference performance.
      </FeatureItem>
      <FeatureItem icon="👨‍💻" title="Developer-Friendly APIs">
        Simple, intuitive APIs in both Python and C++ that hide complexity while providing full
        control. Get started with just a few lines of code, then customize with advanced features as
        needed.
      </FeatureItem>
      <FeatureItem icon="📦" title="Production-Ready Pipelines">
        Pre-built pipelines for text generation, image creation, speech recognition, speech
        generation, and visual language processing. No need to build inference loops or handle
        tokenization - everything works out of the box.
      </FeatureItem>
      <FeatureItem icon="🎨" title="Extensive Model Support">
        Compatible with <Link to="/docs/supported-models">popular models</Link> including Llama,
        Mistral, Phi, Qwen, Stable Diffusion, Flux, Whisper, etc. Easy model conversion from Hugging
        Face and ModelScope.
      </FeatureItem>
      <FeatureItem icon="⚡" title="Lightweight & Efficient">
        Minimal dependencies and smaller disk footprint compared to heavyweight frameworks. Perfect
        for edge deployment, containers, and resource-constrained environments.
      </FeatureItem>
      <FeatureItem icon="🖥️" title="Cross-Platform Compatibility">
        Run the same code on Linux, Windows, and macOS. Deploy across different hardware
        configurations without code changes - from laptops to data center servers.
      </FeatureItem>
    </div>
  </section>
);
