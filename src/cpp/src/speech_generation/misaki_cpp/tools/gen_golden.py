import argparse
import json
from pathlib import Path
import random
import re

from misaki import en

ENGLISH_CASES = [
    {
        "lang": "en",
        "variant": "en-us",
        "text": "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models.",
    },
    {"lang": "en", "variant": "en-us", "text": "[Hello](/həloʊ/)."},
    {"lang": "en", "variant": "en-us", "text": "AIDS"},
    {"lang": "en", "variant": "en-us", "text": "ABTA"},
    {"lang": "en", "variant": "en-us", "text": "A-frame"},
    {"lang": "en", "variant": "en-us", "text": "ASAP"},
    {"lang": "en", "variant": "en-us", "text": "ASCII"},
    {"lang": "en", "variant": "en-us", "text": "is designed for models."},
    {"lang": "en", "variant": "en-us", "text": "G2P engine models."},
    {"lang": "en", "variant": "en-us", "text": "2 + 3"},
    {"lang": "en", "variant": "en-us", "text": "50%"},
    {"lang": "en", "variant": "en-us", "text": "@ home"},
    {"lang": "en", "variant": "en-us", "text": "$12.50"},
    {"lang": "en", "variant": "en-us", "text": "$1"},
    {"lang": "en", "variant": "en-us", "text": "3.14"},
    {"lang": "en", "variant": "en-us", "text": "1st"},
    {"lang": "en", "variant": "en-us", "text": "21st"},
    {"lang": "en", "variant": "en-us", "text": "101st"},
    {"lang": "en", "variant": "en-us", "text": ".50"},
    {"lang": "en", "variant": "en-us", "text": "0.50"},
    {"lang": "en", "variant": "en-us", "text": "1,234"},
    {"lang": "en", "variant": "en-us", "text": "the 7 models"},
    {"lang": "en", "variant": "en-us", "text": "banked 20"},
    {"lang": "en", "variant": "en-us", "text": "1905"},
    {"lang": "en", "variant": "en-us", "text": "2024"},
    {"lang": "en", "variant": "en-us", "text": "1990s"},
    {"lang": "en", "variant": "en-us", "text": "1990's"},
    {"lang": "en", "variant": "en-us", "text": "2000s"},
    {"lang": "en", "variant": "en-us", "text": "12ed"},
    {"lang": "en", "variant": "en-us", "text": "12ing"},
    {"lang": "en", "variant": "en-us", "text": "$0.00"},
    {"lang": "en", "variant": "en-us", "text": "$0.01"},
    {"lang": "en", "variant": "en-us", "text": "$1.01"},
    {"lang": "en", "variant": "en-us", "text": "used to"},
    {"lang": "en", "variant": "en-us", "text": "I used to go"},
    {"lang": "en", "variant": "en-us", "text": "I am here"},
    {"lang": "en", "variant": "en-us", "text": "vs."},
    {"lang": "en", "variant": "en-us", "text": "A vs. B"},
    {"lang": "en", "variant": "en-us", "text": "in town"},
    {"lang": "en", "variant": "en-us", "text": "in"},
    {"lang": "en", "variant": "en-us", "text": "am"},
    {"lang": "en", "variant": "en-us", "text": "by"},
    {"lang": "en", "variant": "en-us", "text": "by far"},
    {"lang": "en", "variant": "en-us", "text": "bye-bye"},
    {"lang": "en", "variant": "en-us", "text": "conduct"},
    {"lang": "en", "variant": "en-us", "text": "to conduct"},
    {"lang": "en", "variant": "en-us", "text": "I conduct"},
    {"lang": "en", "variant": "en-us", "text": "will conduct"},
    {"lang": "en", "variant": "en-us", "text": "contract"},
    {"lang": "en", "variant": "en-us", "text": "to contract"},
    {"lang": "en", "variant": "en-us", "text": "I contract"},
    {"lang": "en", "variant": "en-us", "text": "to conflict"},
    {"lang": "en", "variant": "en-us", "text": "to record"},
]

ENGLISH_GB_CASES = [
    {
        "lang": "en",
        "variant": "en-gb",
        "text": "tomato for models you-all.",
    },
    {"lang": "en", "variant": "en-gb", "text": "AIDS"},
    {"lang": "en", "variant": "en-gb", "text": "A-frame"},
    {"lang": "en", "variant": "en-gb", "text": "ASAP"},
    {"lang": "en", "variant": "en-gb", "text": "ASCII"},
    {"lang": "en", "variant": "en-gb", "text": "is a model"},
    {"lang": "en", "variant": "en-gb", "text": "2 + 3"},
    {"lang": "en", "variant": "en-gb", "text": "50%"},
    {"lang": "en", "variant": "en-gb", "text": "@ home"},
    {"lang": "en", "variant": "en-gb", "text": "$12.50"},
    {"lang": "en", "variant": "en-gb", "text": "$1"},
    {"lang": "en", "variant": "en-gb", "text": "3.14"},
    {"lang": "en", "variant": "en-gb", "text": "1st"},
    {"lang": "en", "variant": "en-gb", "text": "21st"},
    {"lang": "en", "variant": "en-gb", "text": "101st"},
    {"lang": "en", "variant": "en-gb", "text": ".50"},
    {"lang": "en", "variant": "en-gb", "text": "0.50"},
    {"lang": "en", "variant": "en-gb", "text": "1,234"},
    {"lang": "en", "variant": "en-gb", "text": "the 7 models"},
    {"lang": "en", "variant": "en-gb", "text": "banked 20"},
    {"lang": "en", "variant": "en-gb", "text": "1905"},
    {"lang": "en", "variant": "en-gb", "text": "2024"},
    {"lang": "en", "variant": "en-gb", "text": "1990s"},
    {"lang": "en", "variant": "en-gb", "text": "1990's"},
    {"lang": "en", "variant": "en-gb", "text": "2000s"},
    {"lang": "en", "variant": "en-gb", "text": "12ed"},
    {"lang": "en", "variant": "en-gb", "text": "12ing"},
    {"lang": "en", "variant": "en-gb", "text": "$0.00"},
    {"lang": "en", "variant": "en-gb", "text": "$0.01"},
    {"lang": "en", "variant": "en-gb", "text": "$1.01"},
    {"lang": "en", "variant": "en-gb", "text": "used to"},
    {"lang": "en", "variant": "en-gb", "text": "I used to go"},
    {"lang": "en", "variant": "en-gb", "text": "I am here"},
    {"lang": "en", "variant": "en-gb", "text": "vs."},
    {"lang": "en", "variant": "en-gb", "text": "A vs. B"},
    {"lang": "en", "variant": "en-gb", "text": "in town"},
    {"lang": "en", "variant": "en-gb", "text": "in"},
    {"lang": "en", "variant": "en-gb", "text": "am"},
    {"lang": "en", "variant": "en-gb", "text": "by"},
    {"lang": "en", "variant": "en-gb", "text": "by far"},
    {"lang": "en", "variant": "en-gb", "text": "bye-bye"},
    {"lang": "en", "variant": "en-gb", "text": "conduct"},
    {"lang": "en", "variant": "en-gb", "text": "to conduct"},
    {"lang": "en", "variant": "en-gb", "text": "I conduct"},
    {"lang": "en", "variant": "en-gb", "text": "will conduct"},
    {"lang": "en", "variant": "en-gb", "text": "contract"},
    {"lang": "en", "variant": "en-gb", "text": "to contract"},
    {"lang": "en", "variant": "en-gb", "text": "I contract"},
    {"lang": "en", "variant": "en-gb", "text": "to conflict"},
    {"lang": "en", "variant": "en-gb", "text": "to record"},
]

ENGLISH_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*$")


def resolve_data_root() -> Path:
    cpp_root = Path(__file__).resolve().parents[1]
    standalone_data = cpp_root / "data"
    if (
        (standalone_data / "us_gold.json").exists()
        and (standalone_data / "us_silver.json").exists()
        and (standalone_data / "gb_gold.json").exists()
        and (standalone_data / "gb_silver.json").exists()
    ):
        return standalone_data

    monorepo_data = Path(__file__).resolve().parents[2] / "misaki" / "data"
    if (
        (monorepo_data / "us_gold.json").exists()
        and (monorepo_data / "us_silver.json").exists()
        and (monorepo_data / "gb_gold.json").exists()
        and (monorepo_data / "gb_silver.json").exists()
    ):
        return monorepo_data

    raise FileNotFoundError("Could not locate us/gb gold/silver json files in cpp/data or ../misaki/data")


def _extract_pronunciation(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        default = value.get("DEFAULT")
        if isinstance(default, str):
            return default
        for v in value.values():
            if isinstance(v, str):
                return v
    return None


def load_english_lexicon_entries(variant: str) -> dict[str, str]:
    data_root = resolve_data_root()
    prefix = "gb" if variant == "en-gb" else "us"
    with (data_root / f"{prefix}_gold.json").open("r", encoding="utf-8") as f:
        gold_payload = json.load(f)
    with (data_root / f"{prefix}_silver.json").open("r", encoding="utf-8") as f:
        silver_payload = json.load(f)

    entries: dict[str, str] = {}
    for key, value in gold_payload.items():
        pron = _extract_pronunciation(value)
        if pron is not None and ENGLISH_TOKEN_RE.fullmatch(key):
            entries[key] = pron
    for key, value in silver_payload.items():
        if key in entries:
            continue
        pron = _extract_pronunciation(value)
        if pron is not None and ENGLISH_TOKEN_RE.fullmatch(key):
            entries[key] = pron
    return entries


def build_english_cases(sample_size: int, seed: int, variant: str) -> list[dict]:
    base = list(ENGLISH_GB_CASES if variant == "en-gb" else ENGLISH_CASES)
    if sample_size <= 0:
        return base

    base_texts = {c["text"] for c in base}
    entries = load_english_lexicon_entries(variant)
    candidates = []
    for text in entries.keys():
        if text in base_texts:
            continue
        candidates.append(text)

    rng = random.Random(seed)
    sample_count = min(sample_size, len(candidates))
    sampled = rng.sample(candidates, k=sample_count)

    sampled_cases = [{"lang": "en", "variant": variant, "text": text} for text in sampled]
    return base + sampled_cases


def build_engine(lang: str, variant: str):
    if lang == "en":
        british = variant == "en-gb"
        return en.G2P(trf=False, british=british, fallback=None)
    raise ValueError(f"Unsupported lang={lang}, variant={variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/tests/english_golden_cases.jsonl")
    ap.add_argument(
        "--profile",
        default="english",
        choices=["english", "english-gb"],
        help="Corpus profile to generate.",
    )
    ap.add_argument(
        "--english-sample-size",
        type=int,
        default=200,
        help="Number of additional random English lexicon entries to include when --profile english.",
    )
    ap.add_argument(
        "--english-seed",
        type=int,
        default=1337,
        help="Random seed for deterministic English sampling.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.profile == "english":
        selected_cases = build_english_cases(
            args.english_sample_size,
            args.english_seed,
            "en-us",
        )
    elif args.profile == "english-gb":
        selected_cases = build_english_cases(
            args.english_sample_size,
            args.english_seed,
            "en-gb",
        )
    else:
        raise ValueError(f"Unsupported profile: {args.profile}")

    engines: dict[tuple[str, str], object] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for c in selected_cases:
            key = (c["lang"], c["variant"])
            if key not in engines:
                engines[key] = build_engine(c["lang"], c["variant"])
            g2p = engines[key]
            phonemes, _ = g2p(c["text"])
            row = {**c, "phonemes": phonemes}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
