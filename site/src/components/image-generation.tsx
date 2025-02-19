import {SectionImage} from "./Section/section-image"
import {Section} from "@site/src/components/Section";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";
import {LanguageTabs, TabItemCpp, TabItemPython} from "@site/src/components/LanguageTabs";
import CodeBlock from '@theme/CodeBlock';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

const FEATURES = [
    'Alter parameters (width, height, iterations) and compile model for static size',
    'Load LoRA adapters (in safetensor format) and dynamically switch between them',
    'Generate multiple images per one request',
];

const pythonCodeBlock = <CodeBlock language="python">
{`import argparse
from PIL import Image
import openvino_genai

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU, NPU can be used as well
    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)
    image_tensor = pipe.generate(
        args.prompt,
        width=512,
        height=512,
        num_inference_steps=20
    )

    image = Image.fromarray(image_tensor.data[0])
    image.save("image.bmp")`}
</CodeBlock>

const cppCodeBlock = <CodeBlock language="cpp">
{`#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "imwrite.hpp"
int main(int argc, char* argv[]) {

   const std::string models_path = argv[1], prompt = argv[2];
   const std::string device = "CPU";  // GPU, NPU can be used as well

   ov::genai::Text2ImagePipeline pipe(models_path, device);
   ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20));

   imwrite("image.bmp", image, true);
}`}
</CodeBlock>

export const ImageGeneration = () => {
    return (
        <Section.Container>
            <Section.Column>
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
                    <ExploreCodeSamples link={"docs/category/samples"}/>
                    <GoToDocumentation link={"docs/how-to-guides/image-generation"}/>
                </Section.Column>
            </Section.Column>
            <Section.Column>
                <Section.Title>Image generation API</Section.Title>
                <Section.Description>
                    A user-friendly image generation API can be used with generative models to improve creative tools and increase productivity. For instance, it can be utilized in furniture design tools to create various design concepts.
                </Section.Description>
                <SectionImage
                    url={ImagePlaceholder}
                    alt={'Image generation API'}
                />
            </Section.Column>
        </Section.Container>
  )
}
