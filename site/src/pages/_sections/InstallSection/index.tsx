import Button from '@site/src/components/Button';
import Admonition from '@theme/Admonition';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';

import LinuxLogo from '@site/static/img/linux-logo.svg';
import MacOSLogo from '@site/static/img/mac-os-logo.svg';
import WindowsLogo from '@site/static/img/windows-logo.svg';

import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const OS_LIST = [
  {
    title: 'Linux',
    Icon: LinuxLogo,
  },
  {
    title: 'Windows',
    Icon: WindowsLogo,
  },
  {
    title: 'macOS',
    Icon: MacOSLogo,
  },
] as const;

export const InstallSection = () => (
  <section className={styles.installSection}>
    <Heading as="h2" className={styles.sectionTitle}>
      Install OpenVINO™ GenAI
    </Heading>
    <div className={styles.sectionContent}>
      <p className={styles.sectionDescription}>
        Unlock the power of OpenVINO GenAI™ for your projects.
        <br />
        Get started with seamless installation now!
      </p>

      <Button
        label="Install"
        link="/docs/getting-started/installation"
        size="large"
        variant="primary"
      />

      <div className={styles.quickInstall}>
        <Heading as="h3">Quick Installation from PyPi</Heading>
        <CodeBlock language="bash" className={styles.quickInstallCommand}>
          python -m pip install openvino-genai
        </CodeBlock>
      </div>

      <div className={styles.os}>
        <Heading as="h3">Operating Systems</Heading>
        <div className={styles.osList}>
          {OS_LIST.map(({ title, Icon }) => (
            <div key={title} className={styles.osItem}>
              <Icon className={styles.osItemIcon} />
              <span className={styles.osItemTitle}>{title}</span>
            </div>
          ))}
        </div>
      </div>

      <Admonition type="info" title="Need more details?">
        Refer to the <Link to="docs/getting-started/introduction">Getting Started Guide</Link> to
        learn more about OpenVINO GenAI.
      </Admonition>
    </div>
  </section>
);
