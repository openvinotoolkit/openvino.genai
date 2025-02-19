import { ExploreCodeSamples } from "@site/src/components/GoToLink/explore-code-samples";
import { GoToDocumentation } from "@site/src/components/GoToLink/go-to-documentation";
import { Section } from "./Section";

import { LanguageTabs, TabItemCpp, TabItemPython } from "@site/src/components/LanguageTabs";
import CodeBlock from '@theme/CodeBlock';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

// TODO Consider moving to mdx
const FEATURES = [
    'Use different generation parameters (sampling types, etc.)',
    'Optimize for chat scenarios by using chat mode',
    'Load LoRA adapters and dynamically switch between them without recompilation',
    'Use draft model to accelerate generation via Speculative Decoding',
]

const pythonCodeBlock = <CodeBlock language="python">
{`import openvino_genai as ov_genai

# Will run model on CPU, GPU or NPU are possible options
pipe = ov_genai.LLMPipeline("./TinyLlama-1.1B-Chat-v1.0/", "CPU")
print(pipe.generate("The Sun is yellow because", max_new_tokens=100))`}
</CodeBlock>

const cppCodeBlock = <CodeBlock language="cpp">
{`#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");
    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(100)) << '\\n';
}`}
</CodeBlock>

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
                    url={ImagePlaceholder}
                    alt={'Text generation API'}
                />
            </Section.Column>
            <Section.Column>
                <Section.Features features={FEATURES}/>
                <hr/>
                <LanguageTabs>
                    <TabItemPython>
                        {pythonCodeBlock}
                    </TabItemPython>
                    <TabItemCpp>
                        {cppCodeBlock}
                    </TabItemCpp>
                </LanguageTabs>
                <hr/>
                <ExploreCodeSamples link={'docs/category/samples'}/>
                <GoToDocumentation link={'docs/how-to-guides/llm'}/>
            </Section.Column>
        </Section.Container>
    )
}
