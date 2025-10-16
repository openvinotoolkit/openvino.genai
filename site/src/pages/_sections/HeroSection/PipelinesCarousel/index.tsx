import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import clsx from 'clsx';
import { ComponentProps, ComponentType, JSX } from 'react';

import styles from './styles.module.css';

import Carousel from '@site/src/components/Carousel';
import useScreenSize from '@site/src/hooks/use-screen-size';
import ImageIcon from '@site/static/img/image.svg';
import SoundIcon from '@site/static/img/sound-on.svg';
import TextIcon from '@site/static/img/text.svg';

type PipelineItem = {
  title: string;
  Icon: ComponentType<ComponentProps<'svg'>>;
  code: string;
};

const PIPELINES: PipelineItem[] = [
  {
    title: 'Text Generation API',
    Icon: TextIcon,
    code: `ov_pipe = ov_genai.LLMPipeline("TinyLlama")
print(ov_pipe.generate("The Sun is yellow because"))`,
  },
  {
    title: 'Image Generation API',
    Icon: ImageIcon,
    code: `ov_pipe = ov_genai.Text2ImagePipeline("Flux")
image = ov_pipe.generate("Create beautiful Sun")`,
  },
  {
    title: 'Speech Recognition API',
    Icon: SoundIcon,
    code: `ov_pipe = ov_genai.WhisperPipeline("whisper-base")
print(ov_pipe.generate(read_wav("sample.wav")))`,
  },
  {
    title: 'Image Processing API',
    Icon: ImageIcon,
    code: `ov_pipe = ov_genai.VLMPipeline("LLaVA")
print(ov_pipe.generate("Describe images", images))`,
  },
  {
    title: 'Speech Generation API',
    Icon: SoundIcon,
    code: `ov_pipe = ov_genai.Text2SpeechPipeline("speecht5_tts")
result = ov_pipe.generate("Hello OpenVINO GenAI")`,
  },
];

function PipelineExample({ title, Icon, code }: PipelineItem): JSX.Element {
  return (
    <div className={styles.pipelineExample} style={{ flexGrow: '1', padding: '0 0rem' }}>
      <div className={styles.pipelineHeader}>
        <Icon role="img" />
        <Heading as="h3" className={styles.pipelineTitle}>
          {title}
        </Heading>
      </div>
      <div>
        <CodeBlock language="python" className={styles.pipelineCode}>
          {code}
        </CodeBlock>
      </div>
    </div>
  );
}

export default function PipelinesCarousel({ className }: { className: string }): JSX.Element {
  const { width } = useScreenSize();

  const slidesToShow = width < 780 ? 1 : width < 1160 ? 2 : 3;

  return (
    <div className={clsx('container', className)}>
      <Carousel
        slidesToShow={slidesToShow}
        slides={PIPELINES.map((props, idx) => (
          <PipelineExample key={`pipeline-${idx}`} {...props} />
        ))}
      />
    </div>
  );
}
