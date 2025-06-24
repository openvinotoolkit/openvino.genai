import React from 'react';
import styles from './styles.module.css';

import Heading from '@theme/Heading';

type FeatureItemProps = {
  icon: string;
  title: string;
  children: React.ReactNode;
};

export const FeatureItem: React.FC<FeatureItemProps> = ({ icon, title, children }) => (
  <div className={styles.benefitItem}>
    <span className={styles.icon}>{icon}</span>
    <Heading as="h3" className={styles.title}>
      {title}
    </Heading>
    <p className={styles.description}>{children}</p>
  </div>
);
