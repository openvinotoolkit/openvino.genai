import {Section} from "./Section";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {LanguageTabs} from "@site/src/components/LanguageTabs/language-tabs";

const FEATURES = [
    'Use different generation parameters (sampling types, etc.)',
    'Optimize for chat scenarios by using chat mode',
    'Load LoRA adapters and dynamically switch between them without recompilation',
    'Use draft model to accelerate generation via Speculative Decoding',
]

const ITEMS = [
    {
        title: 'Run in C++',
        language: 'cpp',
        content: "#include \"openvino/genai/llm_pipeline.hpp\"\n" +
            "#include <iostream>\n" +
            "\n" +
            "int main(int argc, char* argv[]) {\n" +
            "    std::string models_path = argv[1];\n" +
            "    ov::genai::LLMPipeline pipe(models_path, \"CPU\");\n" +
            "    std::cout << pipe.generate(\"The Sun is yellow because\", ov::genai::max_new_tokens(100)) << '\\n';\n" +
            "}",
    },
    {
        title: 'Run in Python',
        language: 'python',
        content: "import openvino_genai as ov_genai\n" +
            "\n" +
            "# Will run model on CPU, GPU or NPU are possible options\n" +
            "pipe = ov_genai.LLMPipeline(\"./TinyLlama-1.1B-Chat-v1.0/\", \"CPU\")\n" +
            "print(pipe.generate(\"The Sun is yellow because\", max_new_tokens=100))",
    }
]

export const TextGeneration = () => {
    return (
        <Section.Container>
            <Section.Column>
                <Section.Title>Text generation API</Section.Title>
                <Section.Description>
                    An easy-to-use API for text generation can work with an LLM model to create chatbots, AI
                    assistants like financial helpers, and AI tools like legal contract creators.
                </Section.Description>
                <Section.Image
                    url={require('@site/static/img/image-generation-placeholder.webp').default}
                    alt={'Text generation API'}
                />
            </Section.Column>
            <Section.Column>
                <Section.Features features={FEATURES}/>
                <hr/>
                <LanguageTabs items={ITEMS}/>
                <hr/>
                <ExploreCodeSamples link={'docs/category/samples'}/>
                <GoToDocumentation link={'docs/how-to-guides/llm'}/>
            </Section.Column>
        </Section.Container>
    )
}