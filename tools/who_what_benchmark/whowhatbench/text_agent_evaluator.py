# Copyright (C) 2026 Intel Corporation

from typing import Any, Dict, List, Optional

import inspect
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from .registry import BaseEvaluator, register_evaluator
from .utils import get_ignore_parameters_flag, patch_awq_for_inference
from .whowhat_metrics import TextDivergency, TextSimilarity

logger = logging.getLogger(__name__)


@register_evaluator("text-agent")
class TextAgentEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Optional[List[Dict[str, Any]]] = None,
        metrics: str = "similarity",
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens: int = 1024,
        num_samples: Optional[int] = None,
        gen_answer_fn=None,
        empty_adapters: bool = False,
        num_assistant_tokens: int = 0,
        assistant_confidence_threshold: float = 0.0,
        is_genai_backend: bool = False,
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError(
                "Text generation pipeline for evaluation or ground truth data must be defined"
            )

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.generation_fn = gen_answer_fn
        self.empty_adapters = empty_adapters
        self.num_assistant_tokens = num_assistant_tokens
        self.assistant_confidence_threshold = assistant_confidence_threshold
        self.is_genai_backend = is_genai_backend
        self.gt_dir = os.path.dirname(gt_data or "")
        self.target_dir = None

        if base_model:
            result_dir = os.path.join(self.gt_dir, "reference")
            self.gt_data = self._generate_data(base_model, gen_answer_fn, result_dir=result_dir)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when divergency metric is enabled")
            self.divergency = TextDivergency(tokenizer)

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def _read_prompt_value(self, path_or_prompt: Any) -> Optional[str]:
        if isinstance(path_or_prompt, str) and os.path.exists(path_or_prompt):
            with open(path_or_prompt, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def _read_answer_value(self, path_or_answer: Any) -> Optional[str]:
        if isinstance(path_or_answer, str) and os.path.exists(path_or_answer):
            with open(path_or_answer, "r", encoding="utf-8") as f:
                answer = json.load(f)
                if isinstance(answer, str):
                    return answer
                return json.dumps(answer, ensure_ascii=False)
        return None

    def _read_text_data(self, data: pd.DataFrame) -> pd.DataFrame:
        text_data = {"answers": [], "prompts": []}

        for path_or_prompt in data["prompts"].values:
            prompt_value = self._read_prompt_value(path_or_prompt)
            if prompt_value is not None:
                text_data["prompts"].append(prompt_value)
            else:
                text_data["prompts"].append(path_or_prompt)

        for path_or_answer in data["answers"].values:
            answer_value = self._read_answer_value(path_or_answer)
            if answer_value is not None:
                text_data["answers"].append(answer_value)
            else:
                text_data["answers"].append(path_or_answer)

        return pd.DataFrame(text_data)

    @staticmethod
    def _save_prompts_and_answers(prompts, answers, result_dir: str):
        os.makedirs(result_dir, exist_ok=True)
        prompts_dir = os.path.join(result_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)

        prompt_paths = []
        answer_paths = []
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            prompt_path = os.path.join(prompts_dir, f"agent_prompt_{i}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(str(prompt))
            prompt_paths.append(prompt_path)

            answer_path = os.path.join(result_dir, f"agent_output_{i}.json")
            with open(answer_path, "w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=4)
            answer_paths.append(answer_path)

        return prompt_paths, answer_paths

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        output_dir = kwargs.get("output_dir")
        result_folder = os.path.join(output_dir, "target") if output_dir else (
            os.path.join(self.gt_dir, "target") if self.gt_dir else "target"
        )
        self.target_dir = result_folder

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, result_dir=result_folder)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        gt_data_text = self._read_text_data(self.gt_data)
        predictions_text = self._read_text_data(self.predictions)

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                gt_data_text, predictions_text
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                gt_data_text, predictions_text
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions_text["prompts"].values
        self.last_cmp["source_model"] = gt_data_text["answers"].values
        self.last_cmp["optimized_model"] = predictions_text["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        if self.last_cmp is None:
            raise ValueError("No comparison data available. Run score() before worst_examples().")

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        return [row for _, row in res.iterrows()]

    def _get_records(self) -> List[Dict[str, Any]]:
        if self.test_data is None:
            raise ValueError(
                "text-agent evaluator requires a JSON dataset (--dataset) with messages records"
            )
        if not isinstance(self.test_data, list):
            raise ValueError("text-agent evaluator expects test_data as a list of dict records")
        records = self.test_data
        if self.num_samples is not None:
            records = records[: self.num_samples]
        return records

    def _record_max_new_tokens(self, record: Dict[str, Any]) -> int:
        # Keep dataset token budget, but never exceed CLI/global evaluator cap.
        cli_cap = int(self.max_new_tokens)
        record_budget = int(record.get("max_completion_tokens", cli_cap))

        # Guard against invalid non-positive values and keep generation bounded.
        if record_budget <= 0:
            return cli_cap
        if cli_cap <= 0:
            return record_budget

        return min(record_budget, cli_cap)

    @staticmethod
    def _validate_messages_list(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each JSON record must contain a 'messages' list")
        if len(messages) == 0:
            raise ValueError("Each JSON record must contain a non-empty 'messages' list")
        return messages

    def _extract_prompt_preview(self, record: Dict[str, Any]) -> str:
        messages = record.get("messages", [])
        user_texts = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    user_texts.append(content)

        if user_texts:
            return "\n".join(user_texts)

        return ""

    @staticmethod
    def _message_has_content(msg: Dict[str, Any]) -> bool:
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(
                (isinstance(item, dict) and item.get("text", "").strip())
                or (isinstance(item, str) and item.strip())
                for item in content
            )
        return content is not None

    @staticmethod
    def _raise_exception(message: str):
        raise ValueError(message)

    @staticmethod
    def _strftime_now(format_string: str):
        from datetime import datetime

        return datetime.now().strftime(format_string)

    def _get_chat_template_for_model(self, model, tokenizer=None):
        chat_template_map = {
            "qwen3coder": "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_qwen3coder_instruct.jinja",
            "gpt_oss": "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_gpt_oss.jinja",
            "gpt-oss-120b": "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_gpt_oss.jinja",
            "gpt-oss-20b": "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/extras/chat_template_examples/chat_template_gpt_oss.jinja",
        }

        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        if model_type and model_type in chat_template_map:
            template_source = chat_template_map[model_type]
            logger.info(
                "Matched model_type '%s' with chat template source: %s",
                model_type,
                template_source,
            )
            return template_source

        model_dir = getattr(model, "model_dir", None)
        if model_dir:
            model_dir_lower = str(model_dir).lower()
            for key, template_source in chat_template_map.items():
                if key in model_dir_lower:
                    logger.info(
                        "Matched model '%s' with chat template source: %s",
                        key,
                        template_source,
                    )
                    return template_source

        return None

    @staticmethod
    def _get_tokenizer_chat_template(tokenizer) -> Optional[str]:
        if tokenizer is None:
            return None

        if hasattr(tokenizer, "get_chat_template"):
            try:
                return tokenizer.get_chat_template()
            except Exception:
                return getattr(tokenizer, "chat_template", None)

        return getattr(tokenizer, "chat_template", None)

    def _load_chat_template(self, model, tokenizer):
        chat_template_source = self._get_chat_template_for_model(model, tokenizer)
        if chat_template_source:
            if chat_template_source.startswith(("http://", "https://")):
                import requests

                logger.info("Loading chat template from URL: %s", chat_template_source)
                response = requests.get(chat_template_source, timeout=30)
                response.raise_for_status()
                return response.text

            logger.info("Loading chat template from model: %s", chat_template_source)
            custom_tokenizer = AutoTokenizer.from_pretrained(chat_template_source, trust_remote_code=True)
            custom_template = self._get_tokenizer_chat_template(custom_tokenizer)
            if custom_template is not None:
                return custom_template

        if tokenizer is not None:
            try:
                chat_template = self._get_tokenizer_chat_template(tokenizer)
                if chat_template is None:
                    raise ValueError("No chat template found on tokenizer")
                logger.info("Chat template loaded from tokenizer")
                return chat_template
            except Exception:
                logger.warning("Failed to load chat template from tokenizer")

        return None

    def _render_messages_to_prompt(self, messages, tools, chat_template):
        import jinja2

        jinja_env = jinja2.Environment()
        jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
        jinja_env.globals["raise_exception"] = self._raise_exception
        jinja_env.filters["from_json"] = json.loads
        jinja_env.filters["strftime_now"] = self._strftime_now

        chat_template_kwargs = {"add_generation_prompt": True}
        if "enable_thinking" in chat_template:
            chat_template_kwargs["enable_thinking"] = False
        if "reasoning_effort" in chat_template:
            chat_template_kwargs["reasoning_effort"] = "low"

        compiled_template = jinja_env.from_string(chat_template)
        try:
            return compiled_template.render(
                messages=messages,
                tools=tools,
                strftime_now=self._strftime_now,
                **chat_template_kwargs,
            )
        except Exception:
            logger.error("Template render failed. Messages: %s", json.dumps(messages, ensure_ascii=False, default=str))
            logger.error("Template render failed. Tools: %s", json.dumps(tools, ensure_ascii=False, default=str) if tools else None)
            raise

    @staticmethod
    def _apply_tokenizer_chat_template_to_prompt(tokenizer, messages, tools):
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer with apply_chat_template is required for text-agent prompt rendering")

        chat_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools is not None:
            chat_kwargs["tools"] = tools

        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Rendered prompt is empty after chat template application; check/fix the template or messages")
        return prompt

    def _build_prompt_text_without_tools(self, tokenizer, messages, chat_template, backend_name: str) -> str:
        logger.warning(
            "Retrying prompt rendering without tools in %s path due to incompatible tools format",
            backend_name,
        )

        if chat_template is not None:
            try:
                return self._render_messages_to_prompt(messages, None, chat_template)
            except Exception as render_exc:
                logger.warning(
                    "Template rendering without tools failed for current record in %s path; "
                    "falling back to tokenizer.apply_chat_template without tools. Error: %s",
                    backend_name,
                    render_exc,
                )

        return self._apply_tokenizer_chat_template_to_prompt(tokenizer, messages, None)

    def _build_prompt_text(self, tokenizer, messages, tools, chat_template, backend_name: str) -> str:
        if chat_template is not None:
            try:
                return self._render_messages_to_prompt(messages, tools, chat_template)
            except Exception as render_exc:
                logger.warning(
                    "Template rendering failed for current record in %s path; falling back to tokenizer.apply_chat_template. Error: %s",
                    backend_name,
                    render_exc,
                )
        try:
            return self._apply_tokenizer_chat_template_to_prompt(tokenizer, messages, tools)
        except Exception as template_exc:
            if tools is None:
                raise
            logger.warning(
                "tokenizer.apply_chat_template failed with tools in %s path; "
                "retrying without tools. Error: %s",
                backend_name,
                template_exc,
            )
            return self._build_prompt_text_without_tools(tokenizer, messages, chat_template, backend_name)

    def _generate_non_genai(self, model, tokenizer, record: Dict[str, Any], chat_template: Optional[str]) -> str:
        messages = record["messages"]

        tools = record.get("tools")
        device = getattr(model, "device", "cpu")
        prompt = self._build_prompt_text(tokenizer, messages, tools, chat_template, backend_name="HF")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if "input_ids" not in inputs or inputs["input_ids"].shape[-1] == 0:
            raise ValueError("Prompt tokenization produced empty input_ids; prompt cannot be empty")

        if "token_type_ids" in inputs and "token_type_ids" not in list(
            inspect.signature(model.forward).parameters.keys()
        ):
            inputs.pop("token_type_ids")

        is_awq = getattr(model, "is_awq", None) is not None
        generation_kwargs = {
            "do_sample": bool(record.get("temperature", 0.0) > 0),
            "max_new_tokens": self._record_max_new_tokens(record),
            **get_ignore_parameters_flag(),
        }
        temperature = record.get("temperature")
        if temperature is not None:
            generation_kwargs["temperature"] = float(temperature)
        top_p = record.get("top_p")
        if top_p is not None:
            generation_kwargs["top_p"] = float(top_p)

        if is_awq:
            with patch_awq_for_inference(is_awq):
                tokens = model.generate(**inputs, **generation_kwargs)
        else:
            tokens = model.generate(**inputs, **generation_kwargs)

        answer_tokens = tokens[:, inputs["input_ids"].shape[-1]:]
        return tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)[0]

    def _generate_genai(self, model, _tokenizer, record: Dict[str, Any], chat_template: Optional[str]) -> str:
        import openvino_genai

        messages = record["messages"]

        history = openvino_genai.ChatHistory()
        valid_content_messages = 0
        for msg in messages:
            if isinstance(msg, dict):
                history.append(msg)
                if self._message_has_content(msg):
                    valid_content_messages += 1

        if valid_content_messages == 0:
            raise ValueError("All messages are empty in JSON record; prompt cannot be empty")

        kwargs = {
            "do_sample": bool(record.get("temperature", 0.0) > 0),
            "max_new_tokens": self._record_max_new_tokens(record),
            "num_assistant_tokens": int(
                record.get("num_assistant_tokens", self.num_assistant_tokens)
            ),
            "assistant_confidence_threshold": float(
                record.get(
                    "assistant_confidence_threshold",
                    self.assistant_confidence_threshold,
                )
            ),
        }
        temperature = record.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        if self.empty_adapters:
            kwargs["adapters"] = openvino_genai.AdapterConfig()

        tools = record.get("tools")
        prompt = self._build_prompt_text(_tokenizer, messages, tools, chat_template, backend_name="GenAI")

        # Prompt is already rendered with chat template; do not re-apply template in GenAI.
        kwargs["apply_chat_template"] = False
        res = model.generate(prompt, **kwargs)

        if hasattr(res, "texts") and len(res.texts) > 0:
            return res.texts[0]
        return str(res)

    def _generate_data(self, model, gen_answer_fn=None, result_dir=None):
        if gen_answer_fn is None:
            gen_answer_fn = self._generate_genai if self.is_genai_backend else self._generate_non_genai

        chat_template = self._load_chat_template(model, self.tokenizer)

        records = self._get_records()
        prompts = []
        answers = []
        for idx, record in enumerate(tqdm(records, desc="Evaluate pipeline")):
            self._validate_messages_list(record)
            prompt_preview = self._extract_prompt_preview(record)
            if not isinstance(prompt_preview, str) or not prompt_preview.strip():
                raise ValueError(
                    f"Prompt preview is empty for record index {idx}; "
                    "check the messages content in your dataset"
                )
            prompts.append(prompt_preview)
            answers.append(gen_answer_fn(model, self.tokenizer, record, chat_template=chat_template))

        # For long prompts/answers prefer path-based storage in CSV.
        if result_dir:
            prompt_paths, answer_paths = self._save_prompts_and_answers(prompts, answers, result_dir)
            return pd.DataFrame({"prompts": prompt_paths, "answers": answer_paths})

        # Backward-compatible fallback when no result directory is provided.
        return pd.DataFrame({"prompts": prompts, "answers": answers})