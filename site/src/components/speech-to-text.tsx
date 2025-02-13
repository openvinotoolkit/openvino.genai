import {Section} from "@site/src/components/Section";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";
import {LanguageTabs} from "@site/src/components/LanguageTabs/language-tabs";

const FEATURES = [
    'Translate transcription to English',
    'Predict timestamps',
    'Process Long-Form (>30 seconds) audio'
]

const ITEMS = [
    {
        title: 'Run in C++',
        language: 'c',
        content: "#include <iostream>\n" +
            "\n" +
            "#include \"audio_utils.hpp\"\n" +
            "#include \"openvino/genai/whisper_pipeline.hpp\"\n" +
            "\n" +
            "int main(int argc, char* argv[]) {\n" +
            "    std::filesystem::path models_path = argv[1];\n" +
            "    std::string wav_file_path = argv[2];\n" +
            "    std::string device = \"CPU\"; // GPU can be used as well\n" +
            "\n" +
            "    ov::genai::WhisperPipeline pipeline(models_path, device);\n" +
            "\n" +
            "    ov::genai::RawSpeechInput raw_speech = utils::audio::read_wav(wav_file_path);\n" +
            "\n" +
            "    std::cout << pipeline.generate(raw_speech, ov::genai::max_new_tokens(100)) << '\\n';\n" +
            "}"
    },
    {
        title: 'Run in Python',
        language: 'python',
        content: "import openvino_genai\n" +
            "import librosa\n" +
            "\n" +
            "\n" +
            "def read_wav(filepath):\n" +
            "    raw_speech, samplerate = librosa.load(filepath, sr=16000)\n" +
            "    return raw_speech.tolist()\n" +
            "\n" +
            "device = \"CPU\" # GPU can be used as well\n" +
            "pipe = openvino_genai.WhisperPipeline(\"whisper-base\", device)\n" +
            "raw_speech = read_wav(\"sample.wav\")\n" +
            "print(pipe.generate(raw_speech))"
    }
]

export const SpeechToText = () => {
    return (
        <Section.Container>
            <Section.Column>
                <Section.Title>Speech to text API</Section.Title>
                <Section.Description>
                    An intuitive speech-to-text API can work with models like Whisper to enable use cases such as video transcription, enhancing communication tools.
                </Section.Description>
                <Section.Image
                    url={require('@site/static/img/image-generation-placeholder.webp').default}
                    alt={'Speech to text'}
                />
            </Section.Column>
            <Section.Column>
                <Section.Features features={FEATURES} />
                <hr/>
                <LanguageTabs items={ITEMS} />
                <hr/>
                <ExploreCodeSamples link={'docs/how-to-guides/speech-to-text'} />
                <GoToDocumentation link={'https://github.com/openvinotoolkit/openvino.genai'} />
            </Section.Column>
        </Section.Container>

  )
}