#!/usr/bin/env python3
import argparse
import difflib
import os
import random
import re
import codecs
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean


def _build_python_misaki_engine(variant: str):
    from misaki import en

    british = variant == "en-gb"
    return en.G2P(trf=False, british=british, fallback=None, unk="❓")


def _validate_lexicon_data_root(path: str) -> str:
    root = Path(path)
    required = ["us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Lexicon data root '{root}' is missing required files: {', '.join(missing)}")
    return str(root)


def _build_cpp_misaki_engine(variant: str, lexicon_data_root: str | None = None):
    import misaki_cpp_py

    engine = misaki_cpp_py.Engine("en", variant)

    resolved_root = lexicon_data_root or os.getenv("MISAKI_DATA_DIR")
    if resolved_root:
        engine.set_lexicon_data_root(_validate_lexicon_data_root(resolved_root))

    return engine


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    config: str | None
    split: str
    field: str
    weight: float = 1.0

    @property
    def label(self) -> str:
        cfg = self.config if self.config else "default"
        return f"{self.dataset}/{cfg} ({self.split}:{self.field})"


def _profile_specs(profile: str) -> list[DatasetSpec]:
    if profile == "adversarial":
        return [DatasetSpec("wikitext", "wikitext-103-raw-v1", "train", "text", 1.0)]

    if profile == "chatty":
        return [
            DatasetSpec("tweet_eval", "sentiment", "train", "text", 0.45),
            DatasetSpec("glue", "sst2", "train", "sentence", 0.35),
            DatasetSpec("yelp_polarity", None, "train", "text", 0.20),
        ]

    if profile == "realistic":
        return [
            DatasetSpec("xsum", None, "train", "document", 0.50),
            DatasetSpec("ag_news", None, "train", "text", 0.30),
            DatasetSpec("yelp_polarity", None, "train", "text", 0.20),
        ]

    if profile == "mixed":
        return [
            DatasetSpec("xsum", None, "train", "document", 0.35),
            DatasetSpec("ag_news", None, "train", "text", 0.25),
            DatasetSpec("yelp_polarity", None, "train", "text", 0.20),
            DatasetSpec("wikitext", "wikitext-103-raw-v1", "train", "text", 0.20),
        ]

    raise ValueError(f"Unknown profile: {profile}")


def _extract_field_texts(row: dict, field: str) -> list[str]:
    value = row.get(field)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _decode_unicode_escapes(text: str) -> str:
    if "\\u" not in text and "\\U" not in text:
        return text
    try:
        return codecs.decode(text, "unicode_escape")
    except Exception:
        return text


def _normalize_prompt(text: str, decode_escapes: bool) -> str:
    normalized = text
    if decode_escapes:
        normalized = _decode_unicode_escapes(normalized)
    return normalized


def _tokenize_phonemes(phonemes: str) -> list[str]:
    return [token for token in phonemes.split() if token]


def _normalize_phonemes_for_compare(phonemes: str) -> str:
    return phonemes.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def _norm_basic(text: str) -> str:
    text = _normalize_phonemes_for_compare(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text


def _norm_loose(text: str) -> str:
    text = _norm_basic(text)
    for ch in [",", ".", ";", ":", "!", "?"]:
        text = text.replace(ch, " ")
    text = text.replace("==", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _norm_unknown(text: str) -> str:
    return re.sub(r"❓+", "❓", text)


def _maybe_reconfigure_stdout_for_unicode() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def _load_texts(spec: DatasetSpec, max_items: int, seed: int, min_chars: int):
    from datasets import load_dataset

    ds = load_dataset(spec.dataset, spec.config, split=spec.split)
    rng = random.Random(seed)

    candidates = []
    for row in ds:
        for text in _extract_field_texts(row, spec.field):
            normalized = " ".join(text.split())
            if len(normalized) < min_chars:
                continue
            if not re.search(r"[A-Za-z]", normalized):
                continue
            candidates.append(normalized)

    rng.shuffle(candidates)
    return candidates[:max_items]


def _compute_budget(specs: list[DatasetSpec], total: int) -> dict[str, int]:
    budgets: dict[str, int] = {}
    total_weight = sum(max(spec.weight, 0.0) for spec in specs) or 1.0
    assigned = 0

    for spec in specs:
        budget = int(total * max(spec.weight, 0.0) / total_weight)
        budgets[spec.label] = budget
        assigned += budget

    remainder = max(0, total - assigned)
    order = sorted(specs, key=lambda s: s.weight, reverse=True)
    idx = 0
    while remainder > 0 and order:
        budgets[order[idx % len(order)].label] += 1
        idx += 1
        remainder -= 1

    return budgets


def main():
    _maybe_reconfigure_stdout_for_unicode()

    parser = argparse.ArgumentParser(description="Dataset parity runner: misaki_cpp bindings vs Python misaki")
    parser.add_argument(
        "--profile",
        default="single",
        choices=["single", "realistic", "adversarial", "mixed", "chatty"],
        help="Dataset profile. 'single' uses --dataset/--config/--split/--field.",
    )
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name")
    parser.add_argument("--config", default="wikitext-103-raw-v1", help="HF dataset config")
    parser.add_argument("--split", default="train", help="HF split")
    parser.add_argument("--field", default="text", help="Text field name")
    parser.add_argument("--variant", default="en-us", choices=["en-us", "en-gb"], help="English variant")
    parser.add_argument(
        "--lexicon-data-root",
        default=None,
        help="Directory containing us/gb gold/silver lexicon JSON files for misaki_cpp",
    )
    parser.add_argument("--max-items", type=int, default=300, help="Max prompts to evaluate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--min-chars", type=int, default=20, help="Minimum normalized prompt length")
    parser.add_argument(
        "--normalize-input-escapes",
        action="store_true",
        help="Decode \\uXXXX/\\UXXXXXXXX escapes before sending text to both engines",
    )
    parser.add_argument("--show-diffs", type=int, default=10, help="How many worst mismatches to print")
    parser.add_argument(
        "--analyze-categories", action="store_true", help="Show mismatch category breakdown and sample examples"
    )
    parser.add_argument(
        "--analyze-normalization",
        action="store_true",
        help="Show strict/basic/loose/unknown-collapsed normalization match stats",
    )
    parser.add_argument(
        "--analyze-example-limit",
        type=int,
        default=2,
        help="Max examples per mismatch category when --analyze-categories is enabled",
    )
    args = parser.parse_args()

    if args.profile == "single":
        specs = [DatasetSpec(args.dataset, args.config, args.split, args.field, 1.0)]
    else:
        specs = _profile_specs(args.profile)

    budgets = _compute_budget(specs, args.max_items)
    pairs: list[tuple[str, str]] = []
    for idx, spec in enumerate(specs):
        budget = budgets[spec.label]
        if budget <= 0:
            continue
        try:
            part = _load_texts(spec, budget, args.seed + idx, args.min_chars)
        except Exception as ex:
            print(f"[warn] skipped dataset {spec.label}: {ex}")
            continue
        pairs.extend((text, spec.label) for text in part)

    texts = [text for text, _ in pairs]
    if not texts:
        raise RuntimeError("No usable texts were loaded from dataset")

    py_g2p = _build_python_misaki_engine(args.variant)
    cpp_g2p = _build_cpp_misaki_engine(args.variant, args.lexicon_data_root)

    exact = 0
    ratios = []
    token_exact = 0
    token_ratios = []
    mismatches = []
    min_case = None
    skipped = []
    per_source_total = defaultdict(int)
    per_source_exact = defaultdict(int)
    per_source_ratios = defaultdict(list)
    per_source_token_exact = defaultdict(int)
    per_source_token_ratios = defaultdict(list)

    category_mismatch_total = 0
    category_counts = Counter()
    category_examples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    normalization_exact = 0
    normalization_basic = 0
    normalization_loose = 0
    normalization_unknown = 0
    normalization_at_cases = 0
    normalization_at_loose = 0

    def tag_category(name: str, prompt: str, cpp_phonemes: str, py_phonemes: str):
        category_counts[name] += 1
        if len(category_examples[name]) < args.analyze_example_limit:
            category_examples[name].append((prompt, cpp_phonemes, py_phonemes))

    for idx, (text, source_label) in enumerate(pairs):
        per_source_total[source_label] += 1
        input_text = _normalize_prompt(text, args.normalize_input_escapes)
        try:
            py_ps, _ = py_g2p(input_text)
            cpp_ps = cpp_g2p.phonemize(input_text)
        except Exception as ex:
            skipped.append((idx, source_label, text, str(ex)))
            continue

        py_ps_cmp = _normalize_phonemes_for_compare(py_ps)
        cpp_ps_cmp = _normalize_phonemes_for_compare(cpp_ps)

        if args.analyze_normalization:
            if cpp_ps == py_ps:
                normalization_exact += 1
            if _norm_basic(cpp_ps) == _norm_basic(py_ps):
                normalization_basic += 1
            if _norm_loose(cpp_ps) == _norm_loose(py_ps):
                normalization_loose += 1
            if _norm_unknown(cpp_ps) == _norm_unknown(py_ps):
                normalization_unknown += 1
            if "@-@" in input_text or "@,@" in input_text or "@.@" in input_text:
                normalization_at_cases += 1
                if _norm_loose(cpp_ps) == _norm_loose(py_ps):
                    normalization_at_loose += 1

        ratio = difflib.SequenceMatcher(None, cpp_ps_cmp, py_ps_cmp).ratio()
        cpp_tokens = _tokenize_phonemes(cpp_ps_cmp)
        py_tokens = _tokenize_phonemes(py_ps_cmp)
        token_ratio = difflib.SequenceMatcher(None, cpp_tokens, py_tokens).ratio()
        ratios.append(ratio)
        token_ratios.append(token_ratio)
        per_source_ratios[source_label].append(ratio)
        per_source_token_ratios[source_label].append(token_ratio)
        if cpp_ps_cmp == py_ps_cmp:
            exact += 1
            per_source_exact[source_label] += 1
        if cpp_tokens == py_tokens:
            token_exact += 1
            per_source_token_exact[source_label] += 1
        else:
            mismatches.append((ratio, idx, source_label, text, cpp_ps, py_ps))

        if args.analyze_categories and cpp_ps_cmp != py_ps_cmp:
            category_mismatch_total += 1

            if _norm_basic(cpp_ps) == _norm_basic(py_ps):
                tag_category("punct_spacing_only", input_text, cpp_ps, py_ps)

            if "@-@" in input_text or "@,@" in input_text or "@.@" in input_text:
                tag_category("wikimarkup_at_tokens", input_text, cpp_ps, py_ps)

            if re.search(r"[A-Z]+\d|\d+[A-Z]", input_text):
                tag_category("alnum_code_tokens", input_text, cpp_ps, py_ps)

            if re.search(r"\b(No\.|I|II|III|IV|V|VI|VII|VIII|IX|X|[A-Z]{2,})\b", input_text):
                tag_category("roman_or_acronym_context", input_text, cpp_ps, py_ps)

            if re.search(r"[$€£]|\b\d{1,3}(,\d{3})*(\.\d+)?\b", input_text):
                tag_category("numeric_currency_context", input_text, cpp_ps, py_ps)

            if "❓" in cpp_ps or "❓" in py_ps:
                tag_category("contains_unknown_marker", input_text, cpp_ps, py_ps)

            if " at ❓ at " in cpp_ps or "atat" in py_ps:
                tag_category("at_tokenization_pattern", input_text, cpp_ps, py_ps)

            if " nˈO." in cpp_ps or " nˈʌmbəɹ " in py_ps:
                tag_category("abbrev_no_vs_number", input_text, cpp_ps, py_ps)

        if min_case is None or ratio < min_case[0]:
            min_case = (ratio, idx, source_label, text, cpp_ps, py_ps)

    mismatches.sort(key=lambda item: item[0])

    print("== MISAKI CPP VS PYTHON PARITY ==")
    print(f"Profile           : {args.profile}")
    if args.profile == "single":
        print(f"Dataset           : {specs[0].label}")
    else:
        print("Datasets          :")
        for spec in specs:
            print(f"  - {spec.label} (target={budgets.get(spec.label, 0)})")
    print(f"Variant           : {args.variant}")
    print(f"Escapes Normalized: {args.normalize_input_escapes}")
    print(f"Candidate prompts : {len(texts)}")
    print(f"Skipped prompts   : {len(skipped)}")
    print(f"Evaluated prompts : {len(ratios)}")
    if ratios:
        print(f"Exact matches     : {exact}/{len(ratios)} ({100.0 * exact / len(ratios):.2f}%)")
        print(f"Average ratio     : {mean(ratios):.6f}")
        print(f"Minimum ratio     : {min(ratios):.6f}")
        print(f"Token Exact Match : {token_exact}/{len(token_ratios)} ({100.0 * token_exact / len(token_ratios):.2f}%)")
        print(f"Avg Token Ratio   : {mean(token_ratios):.6f}")
    else:
        print("Exact matches     : 0/0 (0.00%)")
        print("Average ratio     : n/a")
        print("Minimum ratio     : n/a")
        print("Token Exact Match : 0/0 (0.00%)")
        print("Avg Token Ratio   : n/a")

    if per_source_total:
        print("\n== PER-DATASET SUMMARY ==")
        for source_label in sorted(per_source_total.keys()):
            total = per_source_total[source_label]
            source_ratios = per_source_ratios[source_label]
            source_token_ratios = per_source_token_ratios[source_label]
            source_exact = per_source_exact[source_label]
            source_token_exact = per_source_token_exact[source_label]
            evaluated = len(source_ratios)
            skipped_count = total - evaluated
            if evaluated > 0:
                print(
                    f"- {source_label}: exact={source_exact}/{evaluated} ({100.0 * source_exact / evaluated:.2f}%), "
                    f"avg_ratio={mean(source_ratios):.6f}, token_exact={source_token_exact}/{evaluated} "
                    f"({100.0 * source_token_exact / evaluated:.2f}%), avg_token_ratio={mean(source_token_ratios):.6f}, "
                    f"skipped={skipped_count}"
                )
            else:
                print(
                    f"- {source_label}: exact=0/0 (0.00%), avg_ratio=n/a, token_exact=0/0 (0.00%), avg_token_ratio=n/a, skipped={skipped_count}"
                )

    if min_case is not None:
        min_ratio, min_idx, min_source_label, min_text, min_cpp_ps, min_py_ps = min_case
        print("\n== MIN-RATIO CASE ==")
        print(f"ratio             : {min_ratio:.6f}")
        print(f"idx/source        : {min_idx} / {min_source_label}")
        print(f"prompt            : {min_text[:500]}")
        print(f"cpp               : {min_cpp_ps[:500]}")
        print(f"py                : {min_py_ps[:500]}")

    if args.analyze_normalization:
        evaluated = len(ratios)
        print("\n== NORMALIZATION ANALYSIS ==")
        if evaluated == 0:
            print("evaluated          : 0")
            print("strict_exact       : 0/0")
            print("basic_norm_match   : 0/0")
            print("loose_norm_match   : 0/0")
            print("unknown_collapsed  : 0/0")
            print("at_cases           : 0")
            print("at_loose_matches   : 0/0")
        else:
            print(f"evaluated          : {evaluated}")
            print(f"strict_exact       : {normalization_exact}/{evaluated}")
            print(f"basic_norm_match   : {normalization_basic}/{evaluated}")
            print(f"loose_norm_match   : {normalization_loose}/{evaluated}")
            print(f"unknown_collapsed  : {normalization_unknown}/{evaluated}")
            print(f"at_cases           : {normalization_at_cases}")
            denom = normalization_at_cases if normalization_at_cases else 1
            print(f"at_loose_matches   : {normalization_at_loose}/{denom}")

    if args.analyze_categories:
        print("\n== MISMATCH CATEGORIES ==")
        print(f"mismatches_analyzed: {category_mismatch_total}")
        if category_mismatch_total == 0:
            print("no mismatches to categorize")
        else:
            for name, count in category_counts.most_common():
                pct = 100.0 * count / category_mismatch_total
                print(f"{name}: {count} ({pct:.1f}% of analyzed mismatches)")

            if category_counts:
                print("\nExamples:")
                for name, _count in category_counts.most_common(6):
                    print(f"[{name}]")
                    for i, (text, cpp_phonemes, py_phonemes) in enumerate(category_examples[name], start=1):
                        print(f"  #{i} text: {text[:120]}")
                        print(f"     cpp : {cpp_phonemes[:140]}")
                        print(f"     py  : {py_phonemes[:140]}")

    if mismatches:
        print("\n== WORST MISMATCHES ==")
        for i, (ratio, idx, source_label, text, cpp_ps, py_ps) in enumerate(mismatches[: args.show_diffs]):
            print(f"[{i}] ratio={ratio:.6f} idx={idx} source={source_label}")
            print(f"  text  : {text[:220]}")
            print(f"  cpp   : {cpp_ps[:220]}")
            print(f"  py    : {py_ps[:220]}")

    if skipped:
        print("\n== SKIPPED PROMPTS (sample) ==")
        for i, (idx, source_label, text, err) in enumerate(skipped[: args.show_diffs]):
            print(f"[{i}] idx={idx} source={source_label}")
            print(f"  text  : {text[:220]}")
            print(f"  error : {err}")


if __name__ == "__main__":
    main()
