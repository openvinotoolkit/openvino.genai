import { ExploreCodeSamples } from '@site/src/components/GoToLink/explore-code-samples';
import { GoToDocumentation } from '@site/src/components/GoToLink/go-to-documentation';
import { LanguageTabs, TabItemCpp, TabItemPython } from '@site/src/components/LanguageTabs';
import { Section } from '@site/src/components/Section';
import CodeBlock from '@theme/CodeBlock';

import ImagePlaceholder from '@site/static/img/image-generation-placeholder.webp';

const FEATURES = [
  'Use different generation parameters (sampling types, etc.)',
  'Optimize for chat scenarios by using chat mode',
  'Pass multiple images to a model',
];

const pythonCodeBlock = (
  <CodeBlock language="python">
    {`import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

# Choose GPU instead of CPU in the line below to run the model on Intel integrated or discrete GPU
pipe = ov_genai.VLMPipeline("./MiniCPM-V-2_6/", "CPU")

image = Image.open("dog.jpg")
image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
image_data = ov.Tensor(image_data)

prompt = "Can you describe the image?"
print(pipe.generate(prompt, image=image_data, max_new_tokens=100))`}
  </CodeBlock>
);

const cppCodeBlock = (
  <CodeBlock language="cpp">
    {`#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::VLMPipeline pipe(models_path, "CPU");
    ov::Tensor rgb = utils::load_image(argv[2]);
    std::cout << pipe.generate(
        prompt,
        ov::genai::image(rgb),
        ov::genai::max_new_tokens(100)
    ) << '\\n';
}`}
  </CodeBlock>
);

export const ImageProcessing = () => {
  return (
    <Section.Container>
      <Section.Column>
        <Section.Title>Image processing with Visual Language Models</Section.Title>
        <Section.Description>
          An easy-to-use API for vision language models can power chatbots, AI assistants like
          medical helpers, and AI tools like legal contract creators.
        </Section.Description>
        <Section.Image
          url={ImagePlaceholder}
          alt={'Image processing with Visual Language Models'}
        />
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
        <GoToDocumentation link="docs/use-cases/image-processing" />
      </Section.Column>
    </Section.Container>
  );
};
