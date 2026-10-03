# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-turn evaluation loop for the tool_call_benchmark."""

import logging

import openvino_genai

from toolcallbench.dataset import load_dataset
from toolcallbench.engine import CaseEngine
from toolcallbench.model_loaders import load_pipeline
from toolcallbench.parser import ToolCallParser, derive_parser_config
from toolcallbench.verdict import compute_verdict

logger = logging.getLogger(__name__)

TURN_TERMINATORS = ("<|im_end|>", "<|endoftext|>", "<|eom_id|>", "<|end_of_text|>")


class ToolCallEvaluator:
    """Runs the benchmark cases against one OpenVINO GenAI pipeline.

    The evaluator drives each case as a multi-turn loop: render the prompt
    with the model's chat template, generate greedily, parse tool calls,
    feed the engine's simulated tool replies back as tool messages.
    """

    def __init__(self, pipeline, tokenizer, max_new_tokens=1024, dataset_path=None):
        """
        :param pipeline: an already-built openvino_genai LLMPipeline or VLMPipeline.
        :param tokenizer: HF tokenizer carrying the model's chat template.
        :param max_new_tokens: generation budget per turn.
        :param dataset_path: optional dataset JSONL; None uses the packaged default.
        """
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.meta, self.cases = load_dataset(dataset_path)
        config = derive_parser_config(tokenizer)
        if "error" in config:
            raise ValueError(f"parser derivation failed: {config['error']}")
        self.parser = ToolCallParser(config)
        self._streamer_chunks = []
        gtok = pipeline.get_tokenizer()

        def on_token(word):
            self._streamer_chunks.append(word)
            return False

        self._streamer = openvino_genai.TextStreamer(
            gtok, on_token, {"skip_special_tokens": False})

    def _generate(self, prompt):
        self._streamer_chunks.clear()
        config = openvino_genai.GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
        )
        try:
            # the prompt is already rendered through the HF chat template
            config.apply_chat_template = False
            config.stop_strings = set(TURN_TERMINATORS)
            if self.tokenizer.eos_token:
                config.stop_strings.add(self.tokenizer.eos_token)
        except Exception:  # pragma: no cover - older builds
            pass
        try:
            self.pipeline.generate(prompt=prompt, generation_config=config,
                                   streamer=self._streamer)
        except TypeError:
            # LLMPipeline names the first argument inputs=
            self.pipeline.generate(inputs=prompt, generation_config=config,
                                   streamer=self._streamer)
        raw = "".join(self._streamer_chunks)
        for stopper in TURN_TERMINATORS:
            pos = raw.find(stopper)
            if pos != -1:
                raw = raw[: pos + len(stopper)]
        return raw

    def run_case(self, case):
        """Run one case end to end.

        :param case: dataset case dict.
        :return: result dict with id, category, kind, correct, bucket,
            format_valid, unsafe, steps, turns.
        """
        engine = CaseEngine(case, self.meta)
        messages = list(case["messages"])
        max_turns = max(engine.max_turns, 3)
        format_valid = True
        turns = 0
        call_seq = 0
        while not engine.done and turns < max_turns + 4:
            prompt = self.tokenizer.apply_chat_template(
                messages, tools=case["tools"], tokenize=False, add_generation_prompt=True)
            bos = getattr(self.tokenizer, "bos_token", None)
            if bos and prompt.startswith(bos):
                prompt = prompt[len(bos):]
            raw = self._generate(prompt)
            turns += 1
            parsed = self.parser.parse(raw)
            if parsed.error in ("malformed", "call_in_thought"):
                format_valid = False
            replies, finished = engine.respond(parsed.calls, text=parsed.text or "")
            if finished:
                break
            # one assistant message per call: llama templates reject multi-call lists
            replies = list(replies) if replies else []
            while len(replies) < len(parsed.calls):
                replies.append("(no output)")
            for call, reply in zip(parsed.calls, replies):
                call_id = f"tcb_{call_seq:06d}"
                call_seq += 1
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": call.get("name", ""),
                                                 "arguments": call.get("arguments", {})}}],
                })
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": str(reply)})
        graded = engine.grade()
        return {
            "id": case["id"],
            "category": case["category"],
            "kind": case["kind"],
            "correct": graded["correct"],
            "bucket": graded["bucket"],
            "format_valid": format_valid,
            "unsafe": graded.get("unsafe", False),
            "steps": graded.get("steps"),
            "turns": turns,
        }

    def evaluate(self, case_ids=None, skip_categories=None):
        """Run the selected cases and produce the full report.

        :param case_ids: optional list of case ids to run; None runs all.
        :param skip_categories: optional list of category names to skip.
        :return: report dict with verdict, results and run configuration.
        """
        skip = set(skip_categories or [])
        known = {c["category"] for c in self.cases}
        unknown = skip - known
        if unknown:
            raise ValueError(f"unknown categories in skip list: {sorted(unknown)}")
        selected = [c for c in self.cases if c["category"] not in skip]
        if case_ids:
            wanted = set(case_ids)
            selected = [c for c in selected if c["id"] in wanted]
        results = []
        for case in selected:
            result = self.run_case(case)
            results.append(result)
            mark = "PASS" if result["correct"] else f"FAIL({result['bucket']})"
            logger.info("%s %s %s (%d turns)", case["id"], case["category"], mark,
                        result["turns"])
        report = compute_verdict(results, total_cases=len(self.cases))
        report["results"] = results
        return report
