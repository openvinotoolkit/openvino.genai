# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import torch
import json
import numpy as np
import logging as log
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
import llm_bench_utils.hook_common as hook_common
from llm_bench_utils.tts_utils import is_kokoro_model_id, normalize_kokoro_lang_code, DEFAULT_KOKORO_VOICE


def set_bf16(model, device, **kwargs):
    try:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            model = model.to(device.lower(), dtype=torch.bfloat16)
            log.info('Set inference precision to bf16')
    except Exception:
        log.error('Catch exception for setting inference precision to bf16.')
        raise RuntimeError('Set prec_bf16 fail.')
    return model


def torch_compile_child_module(model, child_modules, backend='openvino', dynamic=None, options=None):
    if len(child_modules) == 1:
        setattr(model, child_modules[0], torch.compile(getattr(model, child_modules[0]), backend=backend, dynamic=dynamic, fullgraph=True, options=options))
        return model
    setattr(model, child_modules[0], torch_compile_child_module(getattr(model, child_modules[0]), child_modules[1:], backend, dynamic, options))
    return model


def run_torch_compile(model, backend='openvino', dynamic=None, options=None, child_modules=None, memory_data_collector=None):
    if memory_data_collector:
        memory_data_collector.start()
    if backend == 'pytorch':
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        compiled_model = torch.compile(model)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    else:
        log.info(f'Running torch.compile() with {backend} backend')
        start = time.perf_counter()
        if child_modules and len(child_modules) > 0:
            compiled_model = torch_compile_child_module(model, child_modules, backend, dynamic, options)
        else:
            compiled_model = torch.compile(model, backend=backend, dynamic=dynamic, options=options)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    if memory_data_collector:
        memory_data_collector.stop_and_collect_data("compilation")
        memory_data_collector.log_data(compilation=True)
    return compiled_model


def create_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    is_gguf_model = model_path.suffix == '.gguf'
    if not model_path.exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')
    if not is_gguf_model and not (model_path.is_dir() and len(os.listdir(model_path)) != 0):
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    if not device:
        raise RuntimeError('==Failure ==: no device to load')

    log.info(f'Load text model from model path:{model_path}')
    model_class = kwargs['use_case'].pt_cls
    token_class = kwargs['use_case'].tokenizer_cls
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    load_model_kwargs = {"trust_remote_code": False}
    if is_gguf_model:
        load_model_kwargs |= {'gguf_file': str(model_path)}
        model_path = model_path.parent
    try:
        model = model_class.from_pretrained(model_path, **load_model_kwargs)
    except Exception:
        start = time.perf_counter()
        load_model_kwargs['trust_remote_code'] = True
        model = model_class.from_pretrained(model_path, **load_model_kwargs)
    tokenizer = token_class.from_pretrained(model_path, **load_model_kwargs)
    end = time.perf_counter()
    from_pretrain_time = end - start
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data("pretrained")
        memory_data_collector.log_data(compilation=True)

    log.info(f'model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    gptjfclm = 'transformers.models.gptj.modeling_gptj.GPTJForCausalLM'
    lfclm = 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
    bfclm = 'transformers.models.bloom.modeling_bloom.BloomForCausalLM'
    gpt2lmhm = 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'
    gptneoxclm = 'transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM'
    chatglmfcg = 'transformers_modules.pytorch_original.modeling_chatglm.ChatGLMForConditionalGeneration'
    real_base_model_name = str(type(model)).lower()
    log.info(f'Real base model={real_base_model_name}')
    # bfclm will trigger generate crash.

    # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
    if device.upper() == 'GPU':
        device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
    else:
        device = torch.device(device.lower())
    log.info(f'Torch device was set to: {device}')

    if any(x in real_base_model_name for x in [gptjfclm, lfclm, bfclm, gpt2lmhm, gptneoxclm, chatglmfcg]):
        model = set_bf16(model, device, **kwargs)
    else:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            log.info('Param [bf16/prec_bf16] will not work.')
        model.to(device)

    bench_hook = hook_common.get_bench_hook(kwargs['num_beams'], model)

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        dynamic = None
        options = None
        child_modules = None
        if kwargs['torch_compile_dynamic']:
            dynamic = kwargs['torch_compile_dynamic']
        if kwargs['torch_compile_options']:
            options = json.loads(kwargs['torch_compile_options'])
        if kwargs['torch_compile_input_module']:
            child_modules = kwargs['torch_compile_input_module'].split(".")
        compiled_model = run_torch_compile(model, backend, dynamic, options, child_modules, memory_data_collector if kwargs.get("mem_consumption") else None)
        model = compiled_model
    return model, tokenizer, from_pretrain_time, bench_hook, False


def create_image_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_class = kwargs['use_case'].pt_cls
            if kwargs.get("mem_consumption"):
                memory_data_collector.start()
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            pipe = set_bf16(pipe, device, **kwargs)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data("pretrained")
                memory_data_collector.log_data(compilation=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model
    return pipe, from_pretrain_time, False, None


_QWEN3_OMNI_MODEL_TYPES = {
    "qwen3_omni_moe": "Qwen3OmniMoeForConditionalGeneration",
    "qwen3_omni": "Qwen3OmniForConditionalGeneration",
}


def _force_fp32_configs(config):
    # Qwen3-Omni configs mark thinker as bfloat16 but leave talker/code2wav dtype unset.
    # The default heterogeneous mix crashes at matmul time — force fp32 across the whole
    # config tree so matmul dtypes stay consistent.
    stack = [config]
    seen = set()
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        for attr in ("torch_dtype", "dtype"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, "float32")
        for sub_name in (
            "thinker_config",
            "talker_config",
            "code2wav_config",
            "text_config",
            "code_predictor_config",
            "audio_encoder_config",
        ):
            stack.append(getattr(cfg, sub_name, None))


def _load_qwen3_omni_hf_model(model_path):
    import transformers

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    _force_fp32_configs(config)

    model_type = getattr(config, "model_type", "")
    cls_name = _QWEN3_OMNI_MODEL_TYPES.get(model_type)
    model_cls = getattr(transformers, cls_name, None) if cls_name else None
    if model_cls is None:
        raise RuntimeError(
            f"Qwen3-Omni model_type '{model_type}' at {model_path} is not supported by the "
            f"installed transformers build (expected class {cls_name})."
        )

    model = model_cls.from_pretrained(str(model_path), config=config, trust_remote_code=True, dtype=torch.float32)
    # Some Qwen3-Omni submodules (e.g. audio_tower) retain the checkpoint's bfloat16 parameters
    # despite dtype=float32 on from_pretrained, causing dtype-mismatch crashes at matmul/conv.
    # Cast the whole tree to fp32 so parameter dtype is homogeneous.
    model = model.to(torch.float32)
    model.eval()
    return model


_MISSING = object()


class Qwen3OmniPTWrapper:
    """
    HF/PyTorch wrapper adapting Qwen3-Omni's multimodal `generate()` to two runner interfaces:
    the TTS path (`return_audio=True`, returns the `(tokens, waveform)` tuple the runner unpacks)
    and the visual-text path (talker suppressed, tuple unwrapped to just the token-id tensor).
    """

    def __init__(self, model_path):
        self._model = _load_qwen3_omni_hf_model(model_path)
        self.config = self._model.config

    @property
    def generation_config(self):
        return self._model.generation_config

    def to(self, device):
        self._model.to(device)
        return self

    @staticmethod
    def preprocess_inputs(text, image=None, video=None, audio=None, processor=None, **kwargs):
        if processor is None:
            raise ValueError("Processor is required for Qwen3-Omni preprocessing.")

        # extract_prompt_data returns audio from librosa.load as (array, sample_rate); unwrap.
        sampling_rate = None
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            audio = audio[0]
        if isinstance(audio, tuple) and len(audio) == 2:
            audio, sampling_rate = audio

        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})
        if video is not None:
            conversation[0]["content"].insert(0, {"type": "video"})
        if audio is not None:
            conversation[0]["content"].insert(0, {"type": "audio"})

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        processor_kwargs = {}
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = sampling_rate
        return processor(
            images=image, videos=video, text=[prompt], audio=audio, return_tensors="pt", **processor_kwargs
        )

    def generate(self, *args, **kwargs):
        # HF Qwen3-Omni exposes `thinker_*` counterparts instead of the plain generate kwargs;
        # translate so the shared runner paths work unchanged. thinker_eos_token_id has an
        # upstream default of [151645, 151643], so a plain `eos_token_id` in **kwargs is silently
        # dropped by the prefix-strip dispatch loop — translate explicitly (including None, which
        # disables EOS stopping so --infer_count runs to the requested token count).
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        num_beams = kwargs.pop("num_beams", None)
        eos_token_id = kwargs.pop("eos_token_id", _MISSING)
        kwargs.pop("talker_seed", None)
        if max_new_tokens is not None:
            kwargs.setdefault("thinker_max_new_tokens", int(max_new_tokens))
        if num_beams and num_beams > 1:
            kwargs.setdefault("thinker_num_beams", int(num_beams))
        if eos_token_id is not _MISSING:
            kwargs["thinker_eos_token_id"] = eos_token_id

        # Speaker names vary by checkpoint (MoE: Ethan/Chelsie/Aiden/Cherry; dense multilingual:
        # f245/m02/... ). If the requested speaker isn't in this model's speaker_id map, fall back
        # to the first one it does define so the default keeps working across variants.
        speaker = kwargs.get("speaker")
        speaker_id_map = getattr(getattr(self._model.config, "talker_config", None), "speaker_id", None) or {}
        if speaker is not None and speaker_id_map and speaker.lower() not in speaker_id_map:
            fallback = next(iter(speaker_id_map))
            log.warning("Speaker '%s' not available for this checkpoint; falling back to '%s'.", speaker, fallback)
            kwargs["speaker"] = fallback

        # Default to talker-off for the visual-text path: it only consumes token ids and would
        # otherwise pay the cost of audio synthesis (and fail when the checkpoint has no talker).
        # The TTS caller sets return_audio=True explicitly and receives the raw tuple.
        kwargs.setdefault("return_audio", False)

        device = self._model.device
        args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        with torch.no_grad():
            result = self._model.generate(*args, **kwargs)
        # generate() returns (thinker_result, waveform_or_None). When return_audio=False the
        # waveform is None and callers want the token tensor directly; when True the TTS caller
        # unpacks the tuple itself, so pass it through unchanged.
        if kwargs.get("return_audio") is False and isinstance(result, tuple):
            result = result[0]
        return result


def create_image_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f"==Failure ==: model path:{model_path} does not exist")
    if not model_path.is_dir() or len(os.listdir(model_path)) == 0:
        raise RuntimeError(f"==Failure ==: model path:{model_path} is not directory or directory is empty")
    if not device:
        raise RuntimeError("==Failure ==: no device to load")
    if not kwargs.get("is_omni_model", False):
        # Only qwen3-omni currently registers a UseCaseVLM without a pt_cls, so a generic
        # AutoModel load would silently produce a text-only backbone and mismatch preprocess_inputs.
        raise RuntimeError("PyTorch framework for visual_text_gen is only supported for qwen3-omni models.")

    log.info(f"Load Qwen3-Omni HF visual-text model from model path: {model_path}")
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    pipe = Qwen3OmniPTWrapper(model_path)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    from_pretrain_time = time.perf_counter() - start
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data("pretrained")
        memory_data_collector.log_data(compilation=True)
    log.info(f"Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s")

    if device.upper() == "GPU":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        else:
            log.info("CUDA device is unavailable")
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device.lower())
    log.info(f"Torch device was set to: {torch_device}")
    pipe.to(torch_device)

    # Hook the thinker submodule (the actual text-generation loop) so we get first/other-token
    # and per-infer latencies. Qwen3-Omni's top-level generate() delegates token generation to
    # self.thinker.generate(); hooking the outer wrapper would miss the sampling calls entirely.
    bench_hook = hook_common.get_bench_hook(kwargs["num_beams"], pipe._model.thinker)

    # visual_language_generation calls model.preprocess_inputs(..., **processor), so the returned
    # processor mapping must expose the same keys the Optimum path uses: processor / tokenizer / config.
    processor_config = {
        "processor": processor,
        "tokenizer": getattr(processor, "tokenizer", processor),
        "config": pipe.config,
    }
    return pipe, processor_config, from_pretrain_time, bench_hook, False


def create_text_2_speech_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if kwargs.get("is_omni_model", False):
        log.info(f"Load Qwen3-Omni HF model from model path: {model_path}")
        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        start = time.perf_counter()
        pipe = Qwen3OmniPTWrapper(model_path)
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        from_pretrain_time = time.perf_counter() - start
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data("pretrained")
            memory_data_collector.log_data(compilation=True)
        log.info(f"Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s")
        if device:
            if device.upper() == "GPU" and not torch.cuda.is_available():
                log.info("CUDA device is unavailable")
            else:
                torch_device = torch.device("cuda" if device.upper() == "GPU" else device.lower())
                log.info(f"Torch device was set to: {torch_device}")
                pipe.to(torch_device)
        return pipe, processor, None, from_pretrain_time, False
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load text to speech model from model path:{model_path}')
            if kwargs.get("mem_consumption"):
                memory_data_collector.start()
            start = time.perf_counter()
            if is_kokoro_model_id(model_path):
                from kokoro import KPipeline
                from kokoro.model import KModel

                class KokoroPTModelWrapper:
                    def __init__(self, model_dir):
                        self._lang_code = normalize_kokoro_lang_code(kwargs.get("speech_language", ""))
                        config_path = model_dir / "config.json"
                        if config_path.exists():
                            self._kmodel = KModel(config=str(config_path))
                        else:
                            self._kmodel = KModel(repo_id=str(model_dir))
                        self._pipeline = KPipeline(lang_code=self._lang_code, model=self._kmodel)

                    def _update_language(self, language):
                        requested_lang_code = normalize_kokoro_lang_code(language)
                        if requested_lang_code != self._lang_code:
                            self._lang_code = requested_lang_code
                            self._pipeline = KPipeline(lang_code=self._lang_code, model=self._kmodel)

                    def preprocess_input(self, prompt, speaker_embeddings=None, language="", voice=""):
                        self._update_language(language)
                        quiet_pipeline = KPipeline(lang_code=self._lang_code, model=False)

                        selected_voice = voice.strip() if isinstance(voice, str) else ""
                        if not selected_voice:
                            selected_voice = DEFAULT_KOKORO_VOICE

                        # Resolve voice pack during preprocessing to keep timed generation focused on synthesis.
                        voice_or_embedding = (
                            speaker_embeddings
                            if speaker_embeddings is not None
                            else quiet_pipeline.load_voice(selected_voice)
                        )

                        preprocessed_segments = []
                        for result in quiet_pipeline(prompt):
                            phonemes = getattr(result, "phonemes", "")
                            if not phonemes:
                                continue
                            preprocessed_segments.append(
                                {
                                    "phonemes": phonemes,
                                }
                            )

                        if not preprocessed_segments:
                            raise RuntimeError("Kokoro preprocessing produced no valid phoneme segments")

                        return {
                            "segments": preprocessed_segments,
                            "voice_or_embedding": voice_or_embedding,
                        }

                    def generate_from_preprocessed(self, preprocessed_inputs):
                        segments = preprocessed_inputs.get("segments", [])
                        voice_or_embedding = preprocessed_inputs.get("voice_or_embedding")
                        generated_chunks = []

                        for segment in segments:
                            phonemes = segment.get("phonemes")
                            if not phonemes:
                                continue
                            for result in self._pipeline.generate_from_tokens(
                                phonemes,
                                voice=voice_or_embedding,
                                model=self._kmodel,
                            ):
                                if result.audio is not None:
                                    generated_chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))

                        if not generated_chunks:
                            raise RuntimeError("Kokoro generation produced no audio output")

                        return np.concatenate(generated_chunks, axis=0)

                    def generate(self, prompt, speaker_embeddings=None, language="", voice=""):
                        preprocessed = self.preprocess_input(
                            prompt,
                            speaker_embeddings=speaker_embeddings,
                            language=language,
                            voice=voice,
                        )
                        return self.generate_from_preprocessed(preprocessed)

                    def to(self, _device):
                        # Kokoro KPipeline keeps CPU execution internally.
                        return self

                pipe = KokoroPTModelWrapper(model_path)
                processor = None
                vocoder = None
            else:
                model_class = kwargs["use_case"].pt_cls
                token_class = kwargs["use_case"].tokenizer_cls
                pipe = model_class.from_pretrained(model_path)
                vocoder = None
                if kwargs.get("vocoder_path"):
                    vocoder = kwargs["use_case"].vocoder_cls
                pipe = set_bf16(pipe, device, **kwargs)
                processor = token_class.from_pretrained(model_path)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data("pretrained")
                memory_data_collector.log_data(compilation=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model

    return pipe, processor, vocoder, from_pretrain_time, False


def create_ldm_super_resolution_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f'Load image model from model path:{model_path}')
            model_class = kwargs['use_case'].pt_cls
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data("pretrained")
                memory_data_collector.log_data(compilation=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None)
        pipe = compiled_model
    return pipe, from_pretrain_time


def create_text_reranker_model(model_path: Path, device: str, memory_monitor, **kwargs):
    if not model_path.exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')
    if not device:
        raise RuntimeError('==Failure ==: no device to load')
    if not model_path.is_dir() or len(os.listdir(model_path)) == 0:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory or directory is empty')

    log.info(f'Load text reranker model from model path:{model_path}')
    try:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    except Exception:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rerank_use_case = kwargs['use_case']
    rerank_use_case.adjust_model_class_by_config(model_config)
    model_class = rerank_use_case.pt_cls
    token_class = rerank_use_case.tokenizer_cls
    if kwargs.get("mem_consumption"):
        memory_monitor.start()
    start = time.perf_counter()
    pipe = model_class.from_pretrained(model_path)
    pipe = set_bf16(pipe, device, **kwargs)
    end = time.perf_counter()
    if kwargs.get("mem_consumption"):
        memory_monitor.stop_and_collect_data('from_pretrained_phase')
        memory_monitor.log_data('for from pretrained phase')
    from_pretrain_time = end - start
    processor = token_class.from_pretrained(model_path)
    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
    if device.upper() == 'GPU':
        device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
    else:
        device = torch.device(device.lower())
    log.info(f'Torch device was set to: {device}')

    pipe.to(device)

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend, memory_monitor if kwargs.get("mem_consumption") else None)
        pipe = compiled_model

    return pipe, processor, from_pretrain_time, None, False


def create_video_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir() and len(os.listdir(model_path)) != 0:
            log.info(f"Load image model from model path:{model_path}")
            model_class = kwargs["use_case"].pt_cls
            if kwargs.get("mem_consumption"):
                memory_data_collector.start()
            start = time.perf_counter()
            pipe = model_class.from_pretrained(model_path)
            pipe = set_bf16(pipe, device, **kwargs)
            end = time.perf_counter()
            if kwargs.get("mem_consumption"):
                memory_data_collector.stop_and_collect_data("pretrained")
                memory_data_collector.log_data(compilation=True)
            from_pretrain_time = end - start
        else:
            raise RuntimeError(f"==Failure ==: model path:{model_path} is not directory or directory is empty")
    else:
        raise RuntimeError(f"==Failure ==: model path:{model_path} is not exist")

    log.info(f"Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s")

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == "GPU":
            device = torch.device("cuda") if torch.cuda.is_available() else log.info("CUDA device is unavailable")
        else:
            device = torch.device(device.lower())
        log.info(f"Torch device was set to: {device}")

        pipe.to(device)
    else:
        raise RuntimeError("==Failure ==: no device to load")

    if kwargs["torch_compile_backend"]:
        backend = kwargs["torch_compile_backend"]
        compiled_model = run_torch_compile(
            pipe, backend, memory_data_collector if kwargs.get("mem_consumption") else None
        )
        pipe = compiled_model
    tokenizer = None
    bench_hook = None
    use_genai = False
    return pipe, tokenizer, from_pretrain_time, bench_hook, use_genai
