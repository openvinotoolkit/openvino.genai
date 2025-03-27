import argparse
from pathlib import Path
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def main(model_path: str, images_path: str):
    print(f"Selected model: {model_path}\n")
    
    if Path(images_path).is_file():
        image_files = [Path(images_path)]
    else:
        image_files = sorted(
            [f for f in Path(images_path).glob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda x: x.name
        )
    
    if not image_files:
        raise FileNotFoundError(f"No images found in '{images_path}' directory. Supported formats: {IMAGE_EXTENSIONS}")

    images = []
    for file in image_files:
        images.append(
            Image.open(file).convert("RGB")
        )
    
    print("Images:", image_files)

    model = OVModelForVisualCausalLM.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    conversation = [{
        "role": "user",
        "content": [
            *[{"type": "image"} for _ in images],
            {"type": "text", "text": "Describe the images."},
        ],
    }]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    print(prompt)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    result = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    decoded = processor.tokenizer.batch_decode(result[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(decoded)
    with open("ref.txt", "w") as f:
        f.write(f"question:\n{decoded}\n----------\nquestion:\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("-i", "--images_path", type=str, required=True, help="Path to the directory with images.")
    args = parser.parse_args()
    main(args.model_path, args.images_path)
