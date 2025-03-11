import { FC, ReactNode } from 'react';
import styles from './styles.module.css';

interface SectionContainerProps {
  children: ReactNode;
}

export const SectionContainer: FC<SectionContainerProps> = ({ children }) => {
  return <section className={styles.sectionContainer}>{children}</section>;
};
