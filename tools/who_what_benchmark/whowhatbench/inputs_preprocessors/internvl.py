import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class InternVLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": answer})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional[Union["Image", list["Image"]]] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional[Union["VideoInput", list["VideoInput"]]] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=28, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            }
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        if image is not None and not isinstance(image, list):
            image = [image]
        self.update_images(image)

        if image is not None and "<image>" not in text:
            text = "<image>\n" * len(image) + text

        new_message = {"role": "user", "content": text}
        if self.chat_mode:
            self.chat_history.append(new_message)
            messages = self.chat_history
        else:
            messages = [new_message]

        if tokenizer.chat_template is None:
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when chat_template is not defined.")
        else:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = {}
        if self.images is not None:
            if config is None:
                raise ValueError("Config is required.")

            num_patches_list = []
            pixel_values_list = []
            for img in self.images:
                pixel_values = load_image(img, input_size=config.vision_config.image_size)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)

            full_pixel_values = torch.cat(tuple(pixel_values_list), dim=0)
            num_image_token = int(
                (config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio**2)
            )
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
                text = text.replace("<image>", image_tokens, 1)

            inputs.update({"pixel_values": full_pixel_values})
        inputs.update(tokenizer(text, return_tensors="pt"))
        return inputs
