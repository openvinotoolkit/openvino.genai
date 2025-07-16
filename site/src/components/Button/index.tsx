import Link from '@docusaurus/Link';
import clsx from 'clsx';
import React, { CSSProperties } from 'react';

import styles from './styles.module.css';

type ButtonProps = {
  size?: 'sm' | 'lg' | 'small' | 'medium' | 'large' | null;
  outline?: boolean;
  variant?: 'primary' | 'secondary' | 'danger' | 'warning' | 'success' | 'info' | 'link' | string;
  block?: boolean;
  disabled?: boolean;
  className?: string;
  style?: CSSProperties;
  link: string;
  label: string;
  Icon?: React.ComponentType<React.SVGProps<SVGSVGElement>>;
};

export default function Button({
  size = null,
  outline = false,
  variant = 'primary',
  block = false,
  disabled = false,
  className,
  style,
  link,
  label,
  Icon,
}: ButtonProps) {
  const sizeMap = {
    sm: 'sm',
    small: 'sm',
    lg: 'lg',
    large: 'lg',
    medium: null,
  };
  const buttonSize = size ? sizeMap[size] : '';
  const sizeClass = buttonSize ? `button--${buttonSize}` : '';
  const outlineClass = outline ? 'button--outline' : '';
  const variantClass = variant ? `button--${variant}` : '';
  const blockClass = block ? 'button--block' : '';
  const disabledClass = disabled ? 'disabled' : '';
  const destination = disabled ? null : link;

  return (
    <Link className={className} to={destination}>
      <button
        className={clsx('button', sizeClass, outlineClass, variantClass, blockClass, disabledClass)}
        style={style}
        role="button"
        aria-disabled={disabled}
      >
        {Icon && (
          <span className={styles.buttonIcon}>
            <Icon />
          </span>
        )}
        {label}
      </button>
    </Link>
  );
}
