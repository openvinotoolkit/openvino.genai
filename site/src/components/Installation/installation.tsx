import Link from '@docusaurus/Link';
import styles from './styles.module.css';

import Button from '@site/src/components/Button';
import LinuxLogo from '@site/static/img/linux-logo.svg';
import MacOSLogo from '@site/static/img/mac-os-logo.svg';
import WindowsLogo from '@site/static/img/windows-logo.svg';
import Heading from '@theme/Heading';

const getInstallLink = (os: 'WINDOWS' | 'MACOS' | 'LINUX') =>
  `https://docs.openvino.ai/2025/get-started/install-openvino.html?PACKAGE=OPENVINO_GENAI&VERSION=v_2025_0_0&OP_SYSTEM=${os}&DISTRIBUTION=PIP`;

const INSTALLATION_ITEMS = [
  {
    title: 'Linux',
    Icon: LinuxLogo,
    link: getInstallLink('LINUX'),
  },
  {
    title: 'Windows',
    Icon: WindowsLogo,
    link: getInstallLink('WINDOWS'),
  },
  {
    title: 'MacOS',
    Icon: MacOSLogo,
    link: getInstallLink('MACOS'),
  },
];

const InstallationOptions = () => {
  return (
    <div className={styles.installationOptions}>
      {INSTALLATION_ITEMS.map(({ title, Icon, link }) => (
        <Button
          key={title}
          label={title}
          Icon={Icon}
          link={link}
          size="lg"
          outline
          variant="secondary"
        />
      ))}
    </div>
  );
};

export const Installation = () => {
  return (
    <section className={styles.installation}>
      <Heading as="h2" className={styles.installationTitle}>
        Install OpenVINO™ GenAI
      </Heading>
      <p className={styles.installationDescription}>
        Unlock the power of OpenVINO GenAI™ for your projects. <br />
        Get started with seamless installation now!
      </p>

      <InstallationOptions />

      <p>
        Full list of installation options <Link href="docs/overview/installation">here</Link>
      </p>
    </section>
  );
};
