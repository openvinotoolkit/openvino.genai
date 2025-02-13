import styles from './styles.module.css'

const INSTALLATION_ITEMS = [
    {
        title: 'Linux install',
        Icon: require('@site/static/img/linux-logo.svg').default,
    },
    {
        title: 'Windows install',
        Icon: require('@site/static/img/windows-logo.svg').default,
    },
    {
        title: 'MacOS install',
        Icon: require('@site/static/img/mac-os-logo.svg').default,
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
            <h4 className={styles.installationTitle}>Install OpenVINO™ GenAI</h4>
            <p className={styles.installationDescription}>Unlock the power of OpenVINO GenAITM for Your projects. <br/>Get started with seamless installation now!</p>

            <InstallationItems />

            <p>Full list of installation options <a href="docs/overview/installation">here</a></p>
            
        </section>
    )
}