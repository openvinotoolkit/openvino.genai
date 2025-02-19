import { FC, ReactNode } from 'react';
import styles from './styles.module.css';

type SectionColumnProps = {
  children: ReactNode;
  order?: number;
};

export const SectionColumn: FC<SectionColumnProps> = ({ children }) => {
  return <div className={styles.sectionColumn}>{children}</div>;
};
