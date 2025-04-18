import { GoToLink } from '@site/src/components/GoToLink/go-to-link';
import { FC } from 'react';

type ExploreCodeSamplesProps = {
  link: string;
};

export const ExploreCodeSamples: FC<ExploreCodeSamplesProps> = ({ link }) => {
  return <GoToLink link={link} name={'Explore code samples'} />;
};
