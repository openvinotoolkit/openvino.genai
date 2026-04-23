import Heading from '@theme/Heading';
import React from 'react';

import styles from './styles.module.css';

type WithChildren = { children: React.ReactNode };

const Title: React.FC<WithChildren> = ({ children }) => (
  <Heading as="h3" className={styles.title}>
    {children}
  </Heading>
);

const Description: React.FC<WithChildren> = ({ children }) => (
  <p className={styles.description}>{children}</p>
);

const Features: React.FC<WithChildren> = ({ children }) => (
  <div className={styles.features}>
    <strong className={styles.featuresTitle}>Capabilities:</strong>
    <ul className={styles.featuresList}>{children}</ul>
  </div>
);

const Code: React.FC<WithChildren> = ({ children }) => <>{children}</>;

const Actions: React.FC<WithChildren> = ({ children }) => <>{children}</>;

const getChildByType = (
  childrenArray: React.ReactNode[],
  type: React.ElementType
): React.ReactElement => {
  const child = childrenArray.find(
    (child) => React.isValidElement(child) && child.type === type
  ) as React.ReactElement | null;

  if (!child) {
    throw new Error(`UseCaseCard component is missing required child of type ${type}`);
  }

  return child;
};

type UseCaseCardComponents = {
  Title: typeof Title;
  Description: typeof Description;
  Features: typeof Features;
  Code: typeof Code;
  Actions: typeof Actions;
};

const UseCaseCard: React.FC<WithChildren> & UseCaseCardComponents = ({
  children,
}: WithChildren) => {
  const childrenArray = React.Children.toArray(children);

  const titleChild = getChildByType(childrenArray, Title);
  const descriptionChild = getChildByType(childrenArray, Description);
  const featuresChild = getChildByType(childrenArray, Features);
  const codeChild = getChildByType(childrenArray, Code);
  const actionsChild = getChildByType(childrenArray, Actions);

  return (
    <div className={styles.useCaseCard}>
      <div className={styles.header}>{titleChild}</div>

      <div className={styles.content}>
        <div className={styles.contentColumn}>
          {descriptionChild}
          {featuresChild}
        </div>

        <div className={styles.codeColumn}>{codeChild}</div>
      </div>

      <div className={styles.footer}>{actionsChild}</div>
    </div>
  );
};

UseCaseCard.Title = Title;
UseCaseCard.Description = Description;
UseCaseCard.Features = Features;
UseCaseCard.Code = Code;
UseCaseCard.Actions = Actions;

export default UseCaseCard;
