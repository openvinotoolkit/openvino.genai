import clsx from 'clsx';
import React, { useEffect, useRef, useState } from 'react';

import styles from './styles.module.css';

type ChevronIconProps = {
  size?: number;
};

const ChevronLeftIcon: React.FC<ChevronIconProps> = ({ size = 24 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
  >
    <polyline points="15,18 9,12 15,6"></polyline>
  </svg>
);

const ChevronRightIcon: React.FC<ChevronIconProps> = ({ size = 24 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.5"
  >
    <polyline points="9,18 15,12 9,6"></polyline>
  </svg>
);

type CarouselProps = {
  slides: React.ReactNode[];
  autoSlideTimeout?: number;
  enableAutoSlide?: boolean;
  slidesToShow?: number;
};

const Carousel: React.FC<CarouselProps> = ({
  slides = [],
  autoSlideTimeout = 5000,
  enableAutoSlide = true,
  slidesToShow = 1,
}) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const timeoutRef = useRef(null);

  const totalSlides = slides.length;
  const slideWidth = 100 / slidesToShow;
  const maxSlideIndex = Math.max(0, totalSlides - slidesToShow);

  const slidesPages = totalSlides - slidesToShow + 1;

  useEffect(() => {
    if (!enableAutoSlide || isPaused || totalSlides <= slidesToShow) {
      return;
    }

    timeoutRef.current = setTimeout(() => {
      setCurrentSlide((prev) => {
        const nextSlide = prev + 1;
        return nextSlide > maxSlideIndex ? 0 : nextSlide;
      });
    }, autoSlideTimeout);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [
    currentSlide,
    autoSlideTimeout,
    enableAutoSlide,
    isPaused,
    totalSlides,
    slidesToShow,
    maxSlideIndex,
  ]);

  const goToSlide = (index) => {
    const clampedIndex = Math.min(Math.max(0, index), maxSlideIndex);
    setCurrentSlide(clampedIndex);
  };

  const goToPrevious = () => {
    setCurrentSlide((prev) => {
      return prev === 0 ? maxSlideIndex : prev - 1;
    });
  };

  const goToNext = () => {
    setCurrentSlide((prev) => {
      const nextSlide = prev + 1;
      return nextSlide > maxSlideIndex ? 0 : nextSlide;
    });
  };

  const handleMouseEnter = () => {
    setIsPaused(true);
  };

  const handleMouseLeave = () => {
    setIsPaused(false);
  };

  if (!slides || slides.length === 0) {
    return <div>No slides to display</div>;
  }

  return (
    <div
      className={styles.carousel}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className={styles.slidesWrapper}>
        <div
          className={styles.slidesContainer}
          style={{
            transform: `translateX(-${currentSlide * slideWidth}%)`,
          }}
        >
          {slides.map((slide, index) => (
            <div
              key={`carousel-slide-${index}`}
              className={styles.slide}
              style={{ minWidth: `${slideWidth}%` }}
            >
              {slide}
            </div>
          ))}
        </div>
      </div>

      <button
        className={clsx(styles.chevron, styles.chevronLeft)}
        onClick={goToPrevious}
        aria-label="Previous slide"
      >
        <ChevronLeftIcon size={40} />
      </button>

      <button
        className={clsx(styles.chevron, styles.chevronRight)}
        onClick={goToNext}
        aria-label="Next slide"
      >
        <ChevronRightIcon size={40} />
      </button>

      <div className={styles.pagination}>
        {new Array(slidesPages).fill(null).map((_, index) => (
          <button
            key={index}
            className={clsx(styles.dot, currentSlide === index && styles.dotActive)}
            onClick={() => goToSlide(index)}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default Carousel;
