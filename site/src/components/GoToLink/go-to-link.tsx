import styles from "./styles.module.css";

const ChevronRight = require("@site/static/img/chevron-right.svg").default;

interface GoToLinkProps {
    link: string;
    name: string;
}

export const GoToLink = ({ link, name }: GoToLinkProps) => {
    return (
        <a className={styles.goToLink} href={link} target='_blank' rel='noopener noreferrer'>
            {name} <ChevronRight />
        </a>
    )
}
