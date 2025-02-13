import {SectionImage} from "./Section/section-image"
import {Section} from "@site/src/components/Section";
import {LanguageTabs} from "@site/src/components/LanguageTabs/language-tabs";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";

const FEATURES = [
    'Alter parameters (width, height, iterations) and compile model for static size',
    'Load LoRA adapters (in safetensor format) and dynamically switch between them',
    'Generate multiple images per one request',
];

const ITEMS = [
    {
        title: 'Run in C++',
        language: 'c',
        content: "#include \"openvino/genai/image_generation/text2image_pipeline.hpp\"\n" +
            "#include \"imwrite.hpp\"\n" +
            "int main(int argc, char* argv[]) {\n" +
            "\n" +
            "   const std::string models_path = argv[1], prompt = argv[2];\n" +
            "   const std::string device = \"CPU\";  // GPU, NPU can be used as well\n" +
            "\n" +
            "   ov::genai::Text2ImagePipeline pipe(models_path, device);\n" +
            "   ov::Tensor image = pipe.generate(prompt,\n" +
            "        ov::genai::width(512),\n" +
            "        ov::genai::height(512),\n" +
            "        ov::genai::num_inference_steps(20));\n" +
            "\n" +
            "   imwrite(\"image.bmp\", image, true);\n" +
            "}"
    },
    {
        title: 'Run in Python',
        language: 'python',
        content: "import argparse\n" +
            "from PIL import Image\n" +
            "import openvino_genai\n" +
            "\n" +
            "def main():\n" +
            "    parser = argparse.ArgumentParser()\n" +
            "    parser.add_argument('model_dir')\n" +
            "    parser.add_argument('prompt')\n" +
            "    args = parser.parse_args()\n" +
            "\n" +
            "    device = 'CPU'  # GPU, NPU can be used as well\n" +
            "    pipe = openvino_genai.Text2ImagePipeline(args.model_dir, device)\n" +
            "    image_tensor = pipe.generate(\n" +
            "        args.prompt,\n" +
            "        width=512,\n" +
            "        height=512,\n" +
            "        num_inference_steps=20\n" +
            "    )\n" +
            "\n" +
            "    image = Image.fromarray(image_tensor.data[0])\n" +
            "    image.save(\"image.bmp\")"
    }
]

export const ImageGeneration = () => {
    return (
        <Section.Container>
            <Section.Column>
                <Section.Column>
                    <Section.Features features={FEATURES}/>
                    <hr/>
                    <LanguageTabs items={ITEMS}/>
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
                    url={require('@site/static/img/image-generation-placeholder.webp').default}
                    alt={'Image generation API'}
                />
            </Section.Column>
        </Section.Container>
  )
}