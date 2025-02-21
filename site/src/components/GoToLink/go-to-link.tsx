import Link from '@docusaurus/Link';
import styles from './styles.module.css';

import ChevronRight from '@site/static/img/chevron-right.svg';

interface GoToLinkProps {
  link: string;
  name: string;
}

export const GoToLink = ({ link, name }: GoToLinkProps) => {
  return (
    <Link className={styles.goToLink} href={link}>
      {name} <ChevronRight />
    </Link>
  );
};
