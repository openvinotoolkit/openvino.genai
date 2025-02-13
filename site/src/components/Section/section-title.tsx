import {FC, ReactNode} from "react";
import styles from './styles.module.css'

interface SectionTitleProps {
    children: ReactNode;
}

export const SectionTitle: FC<SectionTitleProps> = ({children}) => {
    return (
        <h4 className={styles.sectionTitle}>{children}</h4>
    )
}