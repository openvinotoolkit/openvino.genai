import { FC, ReactNode } from 'react';
import styles from './styles.module.css';

interface SectionDescriptionProps {
  children: ReactNode;
}

export const SectionDescription: FC<SectionDescriptionProps> = ({ children }) => {
  return <p className={styles.sectionDescription}>{children}</p>;
};
