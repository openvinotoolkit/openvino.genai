import styles from './styles.module.css'

import LinuxLogo from '@site/static/img/linux-logo.svg';
import WindowsLogo from '@site/static/img/windows-logo.svg';
import MacOSLogo from '@site/static/img/mac-os-logo.svg';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';

const INSTALLATION_ITEMS = [
    {
        title: 'Linux install',
        Icon: LinuxLogo,
    },
    {
        title: 'Windows install',
        Icon: WindowsLogo,
    },
    {
        title: 'MacOS install',
        Icon: MacOSLogo,
    },
]

const InstallationItems = () => {
    return (
        <ul className={styles.installationOptions}>
            {INSTALLATION_ITEMS.map(({ title, Icon }) => (
                <li key={title} className={styles.installationOptionsItem}>
                    <Icon role='img' aria-label={title} />
                    <span>{title}</span>
                </li>
            ))}
        </ul>
    )
}

export const Installation = () => {
    return (
        <section className={styles.installation}>
            <Heading as='h4' className={styles.installationTitle}>Install OpenVINO™ GenAI</Heading>
            <p className={styles.installationDescription}>Unlock the power of OpenVINO GenAI™ for your projects. <br/>Get started with seamless installation now!</p>

            <InstallationItems />

            <p>Full list of installation options <Link href="docs/overview/installation">here</Link></p>

        </section>
    )
}
