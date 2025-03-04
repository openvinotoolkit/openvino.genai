import Heading from '@theme/Heading';
import { FC, ReactNode } from 'react';
import styles from './styles.module.css';

interface SectionTitleProps {
  children: ReactNode;
}

export const SectionTitle: FC<SectionTitleProps> = ({ children }) => {
  return (
    <Heading as="h4" className={styles.sectionTitle}>
      {children}
    </Heading>
  );
};
