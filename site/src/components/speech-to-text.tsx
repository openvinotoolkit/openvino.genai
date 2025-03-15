import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import { Section } from '@site/src/components/Section';
import CodeBlock from '@theme/CodeBlock';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

const FEATURES = [
  'Translate transcription to English',
  'Predict timestamps',
  'Process Long-Form (>30 seconds) audio',
];

const pythonCodeBlock = (
  <CodeBlock language="python">
    {`import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

device = "CPU" # GPU can be used as well
pipe = openvino_genai.WhisperPipeline("whisper-base", device)
raw_speech = read_wav("sample.wav")
print(pipe.generate(raw_speech))`}
  </CodeBlock>
);

const cppCodeBlock = (
  <CodeBlock language="cpp">
    {`#include <iostream>

#include "audio_utils.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

int main(int argc, char* argv[]) {
    std::filesystem::path models_path = argv[1];
    std::string wav_file_path = argv[2];
    std::string device = "CPU"; // GPU can be used as well

    ov::genai::WhisperPipeline pipeline(models_path, device);

    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);

    std::cout << pipeline.generate(raw_speech, ov::genai::max_new_tokens(100)) << '\\n';
}`}
  </CodeBlock>
);

export const SpeechToText = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Speech to text API</Section.Title>
        <Section.Description>
          An intuitive speech-to-text API can work with models like Whisper to enable use cases such
          as video transcription, enhancing communication tools.
        </Section.Description>
        <Section.Image url={ImagePlaceholder} alt={'Speech to text'} />
      </Section.Column>
      <Section.Column>
        <Section.Features features={FEATURES} />
        <hr />
        <LanguageTabs>
          <TabItemPython>{pythonCodeBlock}</TabItemPython>
          <TabItemCpp>{cppCodeBlock}</TabItemCpp>
        </LanguageTabs>
        <hr />
        <ExploreCodeSamples link="docs/category/samples" />
        <GoToDocumentation link="docs/use-cases/speech-processing" />
      </Section.Column>
    </Section.Container>
  );
};
