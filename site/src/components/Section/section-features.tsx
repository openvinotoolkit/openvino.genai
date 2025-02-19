import Heading from '@theme/Heading';
import { FC, JSX } from 'react';
import styles from './styles.module.css';

interface SectionFeaturesProps {
  features: (string | JSX.Element)[];
}

export const SectionFeatures: FC<SectionFeaturesProps> = ({ features }) => {
  return (
    <div>
      <Heading as="h4" className={styles.sectionFeaturesTitle}>
        Possibilities
      </Heading>

      <ul className={styles.sectionFeaturesList}>
        {features.map((feature, index) => {
          return (
            <li key={index} className={styles.sectionFeaturesListItem}>
              {feature}
            </li>
          );
        })}
      </ul>
    </div>
  );
};
