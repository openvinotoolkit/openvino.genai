import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    __version__,
)
from abc import ABC, abstractmethod
from packaging.version import Version
from typing import TYPE_CHECKING, Optional, Union
import torch
import inspect

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


TRANSFORMERS_VERSION = Version(__version__)


def fix_phi3_v_eos_token_id(model_type, tokenizer):
    """
    phi3_v configs aren't consistent. Override the default
    eos_token_id with the one from a tokenizer similar to
    an example in
    https://huggingface.co/microsoft/Phi-3.5-vision-instruct
    """
    if "phi3_v" == model_type:
        return {"eos_token_id": tokenizer.eos_token_id}
    else:
        return dict()


class VLMInputsPreprocessor(ABC):
    def __init__(self, chat_mode: bool = False):
        self.images = None
        self.videos = None
        self.chat_history = []
        self.chat_mode = chat_mode

    @abstractmethod
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
        return None

    @abstractmethod
    def update_chat_history_with_answer(self, answer):
        pass

    def update_images(self, image):
        if self.chat_mode:
            if image is not None:
                if not self.images:
                    self.images = []
                if isinstance(image, list):
                    self.images.extend(image)
                else:
                    self.images.append(image)
        else:
            self.images = image


class Qwen3VLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

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
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]
        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs


class LLAVAInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if self.chat_mode and getattr(processor, "chat_template", None) is None:
            raise ValueError("Chat template is not set, but pipeline was run in chat mode.")

        if image is not None and not isinstance(image, list):
            image = [image]

        self.update_images(image)

        if getattr(processor, "chat_template", None) is not None:
            templated_prompt = {"role": "user", "content": [{"type": "text", "text": text}]}

            if image is not None:
                templated_prompt["content"].extend([{"type": "image"}] * len(image))

            if self.chat_mode:
                self.chat_history.append(templated_prompt)
                templated_input = self.chat_history
            else:
                templated_input = [templated_prompt]

            prompt = processor.apply_chat_template(templated_input, add_generation_prompt=True, tokenize=False)
        else:
            if image is not None and "<image>" not in text:
                prompt = ("<image>\n") * len(image) + text
            else:
                prompt = text

        if TRANSFORMERS_VERSION > Version("4.47.99") and getattr(processor, "patch_size", None) is None:
            if (
                getattr(config, "vision_config", None) is not None
                and getattr(config.vision_config, "patch_size", None) is not None
            ):
                processor.patch_size = config.vision_config.patch_size
            else:
                raise ValueError(
                    "Processor does not have `patch_size` attribute. Please fix the processor or provide `patch_size` in the config."
                )

        inputs = processor(images=self.images, text=prompt, return_tensors="pt")
        return inputs


class Qwen2VLInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

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
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]

        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

            if self.chat_mode:
                if self.videos is None:
                    self.videos = []
                self.videos.extend(video)
            else:
                self.videos = video
        elif not self.chat_mode:
            self.videos = None

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=self.images,
            text=text_prompt,
            videos=self.videos,
            return_tensors="pt",
        )

        return inputs


class Qwen2_5_VLInputsPreprocessor(Qwen2VLInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)


class Gemma3InputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        content = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            content.extend([{"type": "image"}] * len(image))

        content.append({"type": "text", "text": text})

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            conversation = self.chat_history
        else:
            conversation = [{"role": "user", "content": content}]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # switch off add_bos_token if chat template already includes it
        orig_add_bos_token = processor.tokenizer.add_bos_token
        if getattr(processor.tokenizer, "chat_template", None) and "bos_token" in processor.tokenizer.chat_template:
            processor.tokenizer.add_bos_token = False

        inputs = processor(images=self.images, text=text_prompt, return_tensors="pt")

        # recover add_bos_token flag in tokenizer
        processor.tokenizer.add_bos_token = orig_add_bos_token

        return inputs


class Phi4MMInputsPreprocessor(VLMInputsPreprocessor):
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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")

        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        audio_token = getattr(processor.tokenizer, "audio_token", "<|audio_1|>")

        if audio is not None and audio_token not in text:
            text = audio_token + text

        self.update_images(image)
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, _ in enumerate(image):
                image_token = getattr(processor.tokenizer, "image_token", f"<|image_{i + 1}|>")
                if image_token not in text:
                    text = image_token + text

        text_prompt = ""
        if processor.tokenizer.chat_template is None:
            if self.chat_mode:
                text_hist = ""
                for msg in self.chat_history:
                    if msg["role"] == "user":
                        text_hist += user_prompt + msg["content"] + prompt_suffix
                    elif msg["role"] == "assistant":
                        text_hist += assistant_prompt + msg["content"] + prompt_suffix
                text_prompt = text_hist + assistant_prompt
            else:
                if text.startswith(user_prompt):
                    text_prompt = text
                else:
                    text_prompt = user_prompt + text + prompt_suffix + assistant_prompt
        else:
            if self.chat_mode:
                self.chat_history.append({"role": "user", "content": text})
                conversation = self.chat_history
            else:
                conversation = [{"role": "user", "content": text}]

            text_prompt = processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

        # TODO: audio to chat_mode
        audio_input = {}
        if "audio" in inspect.signature(processor.__call__).parameters:
            sample_rate = None
            if isinstance(audio, tuple):
                audio, sample_rate = audio
            if isinstance(audio, list) and len(audio) == 1 and isinstance(audio[0], tuple):
                audio, sample_rate = audio[0]
            audio_input["audio"] = audio
            if sample_rate is not None:
                audio_input["sampling_rate"] = sample_rate
        else:
            audio_input["audios"] = audio

        inputs = processor(text=text_prompt, images=self.images, **audio_input, return_tensors="pt")
        return inputs


class Phi3MMInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        self.image_offset = 1
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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, _ in enumerate(image):
                image_token = getattr(processor.tokenizer, "image_token", f"<|image_{i + self.image_offset}|>\n")
                if image_token not in text:
                    text = image_token + text
            self.image_offset += len(image)

        if getattr(processor.tokenizer, "chat_template", None) is None:
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when there is no chat_template defined.")
        else:
            new_message = {"role": "user", "content": text}
            if self.chat_mode:
                self.chat_history.append(new_message)
                chat_prompt = self.chat_history
            else:
                chat_prompt = [new_message]

            text = processor.tokenizer.apply_chat_template(chat_prompt, add_generation_prompt=True, tokenize=False)

        inputs = processor(images=self.images, text=text, return_tensors="pt")
        return inputs


class MiniCPMOInputsPreprocessor(VLMInputsPreprocessor):
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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        im_suffix = ""
        if image:
            if not isinstance(image, list):
                image = [image]
            im_suffix = "(<image>./</image>)" * len(image) + "\n"

        new_message = {"role": "user", "content": im_suffix + text}
        if self.chat_mode:
            self.chat_history.append(new_message)
            messages = self.chat_history
        else:
            messages = [new_message]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, self.images, return_tensors="pt")
        inputs.pop("image_sizes", None)

        return inputs


class MiniCPMVInputsPreprocessor(VLMInputsPreprocessor):
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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        im_suffix = ""
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            im_suffix = "(<image>./</image>)" * len(image) + "\n"

        apply_chat_template_func = None
        if getattr(processor, "chat_template", None) is not None:
            apply_chat_template_func = processor.apply_chat_template
        elif getattr(processor.tokenizer, "chat_template", None) is not None:
            apply_chat_template_func = processor.tokenizer.apply_chat_template

        if apply_chat_template_func is not None:
            new_message = {"role": "user", "content": im_suffix + text}
            if self.chat_mode:
                self.chat_history.append(new_message)
                messages = self.chat_history
            else:
                messages = [new_message]

            prompt = apply_chat_template_func(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when there is no chat_template in processor or tokenizer.")
            else:
                prompt = (
                    f"<|im_start|>user\n(<image>./</image>)\n{text}<|im_end|>\n<|im_start|>assistant\n"
                    if image is not None
                    else text
                )

        inputs = processor(prompt, [self.images] if self.images is None else self.images, return_tensors="pt")
        inputs.pop("image_sizes", None)
        return inputs


class LlavaNextInputsPreprocessor(LLAVAInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)


class NanoLlavaInputsPreprocessor(VLMInputsPreprocessor):
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
        if image is not None and processor is None:
            raise ValueError("Processor is required.")

        if not isinstance(image, list):
            image = [image]

        self.update_images(image)
        if len(image) > 0 and image[0] is not None:
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
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_token_id = getattr(config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(config, "image_token_id", -200)

        if "<image>" in text:
            text_chunks = text.split("<image>")
            input_ids = []

            for idx, chunk in enumerate(text_chunks):
                if chunk.strip() != "":
                    chunk_ids = tokenizer(chunk).input_ids
                    input_ids.extend(chunk_ids)
                if idx < len(text_chunks) - 1:
                    input_ids.append(image_token_id)

            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.images is not None:
            result["images"] = processor(images=self.images, return_tensors="pt")["pixel_values"]

        return result


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


# TODO: add support of video models: llava_next_video
MODEL_TYPE_TO_CLS_MAPPING = {
    "qwen3_vl": Qwen3VLInputsPreprocessor,
    "qwen2_vl_text": Qwen2VLInputsPreprocessor,
    "qwen2_vl": Qwen2VLInputsPreprocessor,
    "qwen2_5_vl": Qwen2_5_VLInputsPreprocessor,
    "qwen2_5_vl_text": Qwen2_5_VLInputsPreprocessor,
    "llava": LLAVAInputsPreprocessor,
    "gemma3": Gemma3InputsPreprocessor,
    "phi4mm": Phi4MMInputsPreprocessor,
    "phi4_multimodal": Phi4MMInputsPreprocessor,
    "phi3_v": Phi3MMInputsPreprocessor,
    "minicpmv": MiniCPMVInputsPreprocessor,
    "minicpmo": MiniCPMOInputsPreprocessor,
    "llava_next": LlavaNextInputsPreprocessor,
    "llava-qwen2": NanoLlavaInputsPreprocessor,
    "internvl_chat": InternVLInputsPreprocessor,
}
