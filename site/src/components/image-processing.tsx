import {Section} from "@site/src/components/Section";
import {LanguageTabs} from "@site/src/components/LanguageTabs/language-tabs";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";

const FEATURES = [
    'Use different generation parameters (sampling types, etc.)',
    'Optimize for chat scenarios by using chat mode',
    'Pass multiple images to a model'
]

const ITEMS = [
    {
        title: 'Run in C++',
        language: 'c',
        content: "#include \"load_image.hpp\"\n" +
            "#include <openvino/genai/visual_language/pipeline.hpp>\n" +
            "#include <iostream>\n" +
            "\n" +
            "int main(int argc, char* argv[]) {\n" +
            "    std::string models_path = argv[1];\n" +
            "    ov::genai::VLMPipeline pipe(models_path, \"CPU\");\n" +
            "    ov::Tensor rgb = utils::load_image(argv[2]);\n" +
            "    std::cout << pipe.generate(\n" +
            "        prompt,\n" +
            "        ov::genai::image(rgb),\n" +
            "        ov::genai::max_new_tokens(100)\n" +
            "    ) << '\\n';\n" +
            "}",
    },
    {
        title: 'Run in Python',
        language: 'python',
        content: "import numpy as np\n" +
            "import openvino as ov\n" +
            "import openvino_genai as ov_genai\n" +
            "from PIL import Image\n" +
            "\n" +
            "# Choose GPU instead of CPU in the line below to run the model on Intel integrated or discrete GPU\n" +
            "pipe = ov_genai.VLMPipeline(\"./MiniCPM-V-2_6/\", \"CPU\")\n" +
            "\n" +
            "image = Image.open(\"dog.jpg\")\n" +
            "image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)\n" +
            "image_data = ov.Tensor(image_data)  \n" +
            "\n" +
            "prompt = \"Can you describe the image?\"\n" +
            "print(pipe.generate(prompt, image=image_data, max_new_tokens=100))"
    }
];

export const ImageProcessing = () => {
    return (
        <Section.Container>
            <Section.Column>
                <Section.Features features={FEATURES} />
                <hr/>
                <LanguageTabs items={ITEMS} />
                <hr/>
                <ExploreCodeSamples link={'docs/category/samples'} />
                <GoToDocumentation link={'docs/how-to-guides/vlm'} />
            </Section.Column>
            <Section.Column>
                <Section.Title>
                    Image processing with Visual Language Models
                </Section.Title>
                <Section.Description>
                    An easy-to-use API for vision language models can power chatbots, AI assistants like medical helpers, and AI tools like legal contract creators.
                </Section.Description>
                <Section.Image
                    url={require('@site/static/img/image-generation-placeholder.webp').default}
                    alt={'Image processing with Visual Language Models'}
                />
            </Section.Column>
        </Section.Container>
  )
}