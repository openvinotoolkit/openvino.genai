import argparse
from pathlib import Path
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from openvino.runtime import Core, get_version
import math
from typing import Optional, Tuple, List
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import numpy as np
from transformers import TextStreamer
import time

from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)

def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches

def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid

def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def get_slice_image_placeholder(config, image, tokenizer):
    image_placeholder = (
        tokenizer.im_start
        + tokenizer.unk_token * config.query_num
        + tokenizer.im_end
    )

    slice_images = []

    source_image, patches, best_grid = slice_image(
        image,
        config.max_slice_nums,
        config.scale_resolution,
        config.patch_size,
    )

    slice_images.append(source_image)
    final_placeholder = image_placeholder

    if len(patches) > 0:
        for i in range(len(patches)):
            for j in range(len(patches[0])):
                slice_images.append(patches[i][j])

        final_placeholder += get_grid_placeholder(
            tokenizer, best_grid, config.query_num
        )

    return slice_images, final_placeholder

def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = (
            torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
            + padding_value
        )

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor

def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

class OVMiniCPMForCausalLM(GenerationMixin):
    def __init__(
        self,
        core,
        vision_path,
        resampler_path,
        input_embedding_path,
        language_model_path,
        device,
        tokenizer,
        tm_infer_list,
        vision_infer_list,
    ):
        self.vision = core.compile_model(core.read_model(vision_path), "CPU") #device)
        self.resampler = core.compile_model(core.read_model(resampler_path), device)
        self.input_embeddings = core.compile_model(core.read_model(input_embedding_path), device)
        self.model = core.read_model(language_model_path)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(Path(language_model_path).parent, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = 0
        self.tokenizer = tokenizer
        self.tm_infer_list = tm_infer_list
        self.vision_infer_list = vision_infer_list
        self.max_size = (70, 70)
        self.embed_dim = 2304
        self._pos_embeds = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, 70)).float()

    def _set_2d_pos_cache(self, max_size):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self._pos_embed = pos_embed

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size)

    def resample(self, x, tgt_sizes):
        bs = x.shape[0]

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self._pos_embeds[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        res = torch.from_numpy(self.resampler([x, pos_embed, key_padding_mask])[0])
        return res


    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_sizes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            pixel_values,
            attention_mask,
            past_key_values,
            position_ids,
            image_sizes,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_sizes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""

        inputs = {}
        if past_key_values is not None:
            inputs = {}
            if not self.stateful:
                past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))
            # input_ids = np.array(input_ids)[:, -1:]

            inputs_embeds = self.input_embeddings(input_ids)[0] * self.config.scale_emb
            inputs["inputs_embeds"] = inputs_embeds
            batch_size = input_ids.shape[0]

            if "beam_idx" in self.input_names:
                inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

            target_length = input_ids.shape[1]
            past_length = self.past_len
            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            inputs["attention_mask"] = attention_mask
            inputs["position_ids"] = position_ids
        else:
            inputs = self.prepare_multimodal_input(input_ids, pixel_values, attention_mask, position_ids, image_sizes)

        # Run inference
        start = time.perf_counter()
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        tm_infer_list.append(generation_time)

        logits = torch.from_numpy(self.request.get_tensor(self.output_names[0]).data)

        if not self.stateful:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = ((),)
        self.past_len += inputs["inputs_embeds"].shape[1]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prepare_multimodal_input(self, input_ids, pixel_values, attention_mask, position_ids, image_sizes=None):
        """Preprocessing function for embedding multimodal data"""
        tm_infer_list.clear()
        vision_infer_list.clear()
        inputs = {}

        start = time.perf_counter()

        batch_size = input_ids.shape[0]
        if not self.stateful:
            for input_name in self.key_value_input_names:
                model_inputs = self.modeget_anyres_image_grid_shapel.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
        else:
            self.past_len = 0
            self.request.reset_state()
            # Set initial value for the next beam_idx input that will be used at the current iteration
            # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
            self.next_beam_idx = np.arange(batch_size, dtype=int)

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[1]
        # skip im_start
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[1]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bound = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )

        vision_input = {}
        vision_input["input_ids"] = input_ids #.unsqueeze(0)
        vision_input["image_bound"] = [image_bound]
        vision_input["pixel_values"] = pixel_values

        (inputs["inputs_embeds"], vision_hidden_states, ) = self.get_vllm_embedding(vision_input)

        if pixel_values is None:
            inputs["attention_mask"] = attention_mask
            if position_ids is None:
                position_ids = torch.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            inputs["position_ids"] = position_ids

        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids

        end = time.perf_counter()

        prepareinput_time = (end - start) * 1000
        tm_infer_list.append(prepareinput_time)

        return inputs

    def get_vision_embedding(self, pixel_values):
        res = []
        index = 0
        dtype = torch.float32
        for pixel_value in pixel_values:
            H, W = pixel_value.shape[-2:]
            tgt_size = torch.tensor([[math.ceil(H / self.config.patch_size), math.ceil(W / self.config.patch_size)]])
            start = time.perf_counter()
            vision_embedding = self.vision(pixel_value.unsqueeze(0).type(dtype))
            end = time.perf_counter()
            vision_infer_list.append(((end - start) * 1000))
            image_features = torch.from_numpy(vision_embedding[0])

            start = time.perf_counter()
            resampler_feat = self.resample(image_features, tgt_size)
            end = time.perf_counter()
            vision_infer_list.append(((end - start) * 1000))

            resampler_feat = resampler_feat[0]

            res.append(resampler_feat)
            index = index + 1
        return torch.vstack(res)

    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                else:
                    vision_hidden_states.append([])

        else:
            vision_hidden_states = data["vision_hidden_states"]

        embedding_token = torch.from_numpy(self.input_embeddings(data["input_ids"])[0])
        vllm_embedding = (embedding_token * self.config.scale_emb)

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            #print("ov cur_vs_hs ", cur_vs_hs)
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if not self.stateful:
                cache_length = past_length = past_key_values[0][0].shape[2]
            else:
                cache_length = past_length = self.past_len

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.llava
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch gllavaenerationsubset_siz
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

class preprocessor:
    def __init__(self, config, tokenizer):
        """
        Initialization class parameters
        """
        self.config = config
        self.tokenizer = tokenizer
        self.transform = self.init_transform()

    def init_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    def _convert_to_tensors(
        self, tokenizer, input_str, max_inp_length: Optional[int] = None
    ):
        if tokenizer.add_bos_token:
            input_ids = tokenizer.encode(input_str)
        else:
            input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        model_input = {}
        model_input["input_ids"] = input_ids.unsqueeze(0)

        return model_input

    def _process_list(
        self, tokenizer, data_list: List[str], max_inp_length: Optional[int] = None
    ):
        pad_keys = ["input_ids"]
        input_tensors = []
        for data in data_list:
            input_tensors.append(
                self._convert_to_tensors(tokenizer, data, max_inp_length)
            )
        padded = {}
        for key in pad_keys:
            padded[key] = pad(input_tensors, key, padding_side="left")
        return padded

    def get_inputs(self, image:torch.Tensor, msgs = None, max_inp_length=2048, **generate_kwargs):
        """
        use the model as a conditional generator
        Parameters:
            image (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            msgs ('')
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt
        prompt = ""
        for i, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
                if config.slice_mode:
                    images, final_placeholder = get_slice_image_placeholder(config, image, tokenizer)
                    content = final_placeholder + "\n" + content
                else:
                    images = [image]
                    content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * config.query_num
                        + tokenizer.im_end
                            + "\n"
                            + content
                        )
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
        prompt += "<AI>"
        final_input = prompt

        data_list=[final_input]
        img_list=[images]

        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length)
        pixel_values = []
        for i in range(bs):
            img_inps = []
            for img in img_list[i]:
                img_inps.append(self.transform(img))
            if img_inps:
                pixel_values.append(img_inps)
            else:
                pixel_values.append([])
        model_inputs["pixel_values"] = pixel_values

        return model_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export minicpm-v2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument('-d', '--device', default='cpu', help='inference device')
    parser.add_argument('-pic', '--picture', help='picture file')
    parser.add_argument('-p', '--prompt', default=None, help='one prompt')

    args = parser.parse_args()
    model_path = args.model_id
    device = args.device.upper()

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    image = Image.open(args.picture).convert('RGB')
    question = args.prompt #'描述画面内容' #'What is in the image?'
    msgs = [{'role': 'user', 'content': question}]

    #image resize
    image_width=800
    image_height=800
    image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)

    #vision info
    config.patch_size= 14
    config.num_prefix_tokens = 0

    VISION_MODEL_OV = Path(f"{model_path}/openvino_vision.xml")
    RESAMPLER_MODEL_OV = Path(f"{model_path}/openvino_resampler.xml")
    EBEDDING_MODEL_OV = Path(f"{model_path}/openvino_embedding.xml")
    LANGUAGE_MODEL_OV = Path(f"{model_path}/openvino_model.xml")

    tm_infer_list = []
    vision_infer_list = []

    # create OpenVINO Core object instance
    core = ov.Core()

    version = get_version()
    print("OpenVINO version \n", version)

    #set cache
    core.set_property({'CACHE_DIR': "minicpmv2_cache"})

    ov_minicpmv_model = OVMiniCPMForCausalLM(core, VISION_MODEL_OV, RESAMPLER_MODEL_OV, EBEDDING_MODEL_OV, LANGUAGE_MODEL_OV, device, tokenizer=tokenizer, tm_infer_list=tm_infer_list, vision_infer_list=vision_infer_list)

    preprocessor = preprocessor(config=config, tokenizer=tokenizer)
    inputs = preprocessor.get_inputs(image=image, msgs=msgs)

    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    output = ov_minicpmv_model.generate(**inputs, max_new_tokens=2048, streamer=streamer)

    if len(tm_infer_list) > 2:
        avg_token = sum(tm_infer_list[2:]) / (len(tm_infer_list) - 2)
        print(f"warm up Inputs len {inputs['input_ids'].shape[1]} Vision latency: {tm_infer_list[0]:.2f} ms, First token latency: {tm_infer_list[1]:.2f} ms, Output len {len(tm_infer_list) - 1}, Avage token latency: {avg_token:.2f} ms")

    if len(vision_infer_list) > 2:
        sum_vision = sum(vision_infer_list[::2])
        sum_sampler = sum(vision_infer_list[1: (len(vision_infer_list) -1) : 2])
        print(f"vision latency : {sum_vision:.2f} ms, sampler latency : {sum_sampler:.2f} ms")
