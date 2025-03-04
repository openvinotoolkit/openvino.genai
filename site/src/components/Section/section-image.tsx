import { CSSProperties } from 'react';

interface ImageProps {
  url: string;
  alt: string;
  style?: CSSProperties;
}

export const SectionImage = ({ url, style, alt }: ImageProps) => {
  return <img src={url} style={{ aspectRatio: 1, width: '100%', ...style }} alt={alt} />;
};
