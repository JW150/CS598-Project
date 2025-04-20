from __future__ import annotations

import argparse
import pickle
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import nltk
import pandas as pd
import swifter

pd.options.mode.chained_assignment = None

from preprocess.char_mapping import *
from preprocess.patterns import *
from preprocess.utils import *
from preprocess.chart_process import get_instructions, get_hospital_course, strip_short_text, why_waht_next_process, remove_regex_dict


@dataclass
class CLIConfig:
    input_file: str
    output_dir: str
    start_from_step: int = 0
    reproduce_avs_extraction: bool = False

    @staticmethod
    def from_cli() -> "CLIConfig":
        parser = argparse.ArgumentParser(description="Pre‑process MIMIC discharge summaries.")
        parser.add_argument("--input_file", required=True, help="Path to input CSV or pickle file")
        parser.add_argument("--output_dir", required=True, help="Directory for processed output")
        parser.add_argument("--start_from_step", type=int, default=0, help="Resume pipeline from this step id")
        parser.add_argument("--reproduce_avs_extraction", action="store_true", help="Run only the AVS extraction branch")
        args = parser.parse_args()
        return CLIConfig(**vars(args))


StepFn = Callable[[pd.DataFrame], pd.DataFrame]
STEP_REGISTRY: Dict[int, StepFn] = {}

def _register(step_id: int) -> Callable[[StepFn], StepFn]:
    def _decorator(func: StepFn) -> StepFn:
        STEP_REGISTRY[step_id] = func
        return func
    return _decorator


@_register(1)
def step_dedupe_and_service(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 1] Deduplicate + extract service tags …")
    df = df.sort_values(["subject_id", "hadm_id"], ascending=[False, False], ignore_index=True)

    before = len(df)
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates(subset=["subject_id", "hadm_id", "text"], keep="first")
    print(f"    Removed {before - len(df)} exact duplicates.")

    before = len(df)
    df = df.drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")
    print(f"    Kept most-recent note per hospital stay - removed {before - len(df)} others.")

    re_service = re.compile(r"^Service: (.*)$", re.IGNORECASE | re.MULTILINE)
    re_service_alt = re.compile(r"^Date of Birth:.*Sex:\s{0,10}\w\s{0,10}___: (.*)$", re.IGNORECASE | re.MULTILINE)

    df["service"] = (
        df["text"].swifter.apply(lambda s: re_service.search(s).group(1) if re_service.search(s) else None) 
    )
    mask_na = ~df["service"].notnull()
    df.loc[mask_na, "service"] = (
        df.loc[mask_na, "text"].swifter.apply(lambda s: re_service_alt.search(s).group(1) if re_service_alt.search(s) else None)
    )
    df["service"] = df["service"].fillna("").str.strip().str.strip(string.punctuation)
    df["service"] = df["service"].apply(lambda s: SERVICE_MAPPING.get(s, s))

    print("    Service label distribution:")
    print(Counter(df["service"]).most_common(20))

    df["text"] = df["text"].replace(SPECIAL_CHARS_MAPPING_TO_ASCII, regex=True)
    return df


@_register(2)
def step_split_note_and_summary(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 2] Split discharge summaries from note body …")
    discharge_re = re.compile(r"Discharge Instructions:\n", re.IGNORECASE)
    mask = df["text"].str.contains(discharge_re, regex=True)
    removed = len(df) - mask.sum()
    df = df[mask]

    df[["hospital_course", "summary"]] = df.apply(lambda x: discharge_re.split(x["text"], 1), axis=1, result_type="expand")
    df["hospital_course"] = df["hospital_course"].str.strip()
    df["summary"] = df["summary"].str.strip()
    df["original_summary"] = df["summary"]
    df["brief_hospital_course"] = df["hospital_course"].apply(get_hospital_course)

    for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
        df["summary"] = df["summary"].str.replace(k, v, regex=False)

    df = strip_short_text(df)
    print(f"    Removed {removed} rows without a Discharge Instructions section.")
    return df


@_register(3)
def step_truncate_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove common ‘header’ fragments that precede real discharge instructions."""
    print("\n[Step 3] Trim leading boiler‑plate in summaries …")

    df["summary"] = (
        df["summary"]
        .str.strip()
        .apply(lambda s: re_multiple_whitespace.sub(" ", s))
        .apply(lambda s: re_line_punctuation_wo_underscore.sub("", s))
    )

    post_proc = lambda s: re_ds_punctuation_wo_underscore.sub("", s.strip())
    df["summary"] = df["summary"].apply(post_proc)

    df = remove_regex_dict(
        df,
        UNNECESSARY_SUMMARY_PREFIXES,
        postprocess=post_proc,
        keep=1,
    )

    df = strip_short_text(df)
    return df


@_register(4)
def step_remove_static_patterns(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4] Re‑phrase static list templates and deidentification …")

    df["summary"] = df["summary"].apply(lambda s: "\n".join(x.strip() for x in s.split("\n")))
    df["summary"] = df["summary"].apply(lambda s: re_line_punctuation_wo_fs.sub("", s))
    df["summary"] = df["summary"].apply(lambda s: re_fullstop.sub("", s))
    df["summary"] = df["summary"].apply(lambda s: re_multiple_whitespace.sub(" ", s))

    df["summary"] = why_waht_next_process(df["summary"])

    headings_re = re.compile(WHY_WHAT_NEXT_HEADINGS, re.MULTILINE | re.IGNORECASE)
    df["summary"] = df["summary"].apply(lambda s: headings_re.sub("\n", s))

    df["summary"] = df["summary"].apply(lambda s: re_newline_in_text.sub(" ", s))
    df["summary"] = df["summary"].apply(lambda s: re_multiple_whitespace.sub(" ", s))

    for replacement, regex in SIMPLE_DEIDENTIFICATION_PATTERNS:
        count = df["summary"].apply(lambda s: len(regex.findall(s))).sum()
        print(f"    Replaced {count} matches with '{replacement}'.")
        df["summary"] = df["summary"].apply(lambda s, _r=regex, _rep=replacement: _r.sub(_rep, s))

    df = strip_short_text(df)
    return df


@_register(5)
def step_truncate_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 5] Trim trailing boiler‑plate …")

    post_proc = lambda s: re_multiple_whitespace.sub(" ", s.strip()) 
    df["summary"] = df["summary"].apply(post_proc)

    df = remove_regex_dict(
        df,
        RE_SUFFIXES_DICT,
        postprocess=post_proc,
        keep=0,
    )

    df["summary"] = df["summary"].apply(
        lambda s: re_incomplete_sentence_at_end.split(s, 1)[0]
    )

    re_no_text = re.compile(r"^[^a-z_\n]+$", re.IGNORECASE | re.MULTILINE)
    df["summary"] = df["summary"].apply(lambda s: re_no_text.sub("", s))

    re_item_start = re.compile(r"^" + re_item_element, re.MULTILINE)
    df["summary"] = df["summary"].apply(lambda s: re_item_start.sub("", s))

    df = strip_short_text(df)
    return df


@_register(6)
def step_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 6] Statistical quality gates …")
    min_chars = 350
    max_paragraphs = 5
    min_sentences = 3
    ratio_deid = 10

    before = len(df)
    df = df[df["summary"].str.len() >= min_chars]
    print(f"    Length filter removed {before - len(df)} rows (<{min_chars} chars).")

    df["sentences"] = df["summary"].swifter.apply(lambda s: nltk.sent_tokenize(s))
    before = len(df)
    df = df[df["sentences"].map(len) >= min_sentences]
    print(f"    Sentence count filter removed {before - len(df)} rows (<{min_sentences} sentences).")

    before = len(df)
    df = df[df["summary"].map(lambda s: s.count("\n\n")) <= max_paragraphs]
    print(f"    Paragraph filter removed {before - len(df)} rows (>{max_paragraphs} paragraphs).")

    df["summary"] = df["sentences"].swifter.apply(lambda s: re_whitespace.sub(" ", " ".join(s)))
    df.drop(columns=["sentences"], inplace=True)

    for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
        df["summary"] = df["summary"].str.replace(v, k, regex=False)

    df["num_deidentified"] = df["summary"].swifter.apply(lambda s: s.count("___"))
    before = len(df)
    df = df[df["num_deidentified"] <= df["summary"].map(lambda s: len(s.split(" ")) / ratio_deid)]
    print(f"    Heavy de‑identification filter removed {before - len(df)} rows.")

    return df


@_register(7)
def step_validate_hospital_course(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 7] Validate presence and length of hospital courses …")
    min_bhc_chars = 500
    before = len(df)
    df = df[df["hospital_course"].notnull()]
    print(f"    {before - len(df)} rows lacked a hospital_course section.")

    before = len(df)
    df = df[df["brief_hospital_course"].notnull()]
    print(f"    {before - len(df)} rows lacked a brief_hospital_course.")

    triple_nl = re.compile(r"\n{3,}", re.MULTILINE)
    df["hospital_course"] = df["hospital_course"].apply(lambda s: triple_nl.sub("\n\n", s))
    df["brief_hospital_course"] = df["brief_hospital_course"].apply(lambda s: triple_nl.sub("\n\n", s))

    before = len(df)
    df = df[df["brief_hospital_course"].str.len() >= min_bhc_chars]
    print(f"    {before - len(df)} rows removed (<{min_bhc_chars} chars brief hospital course).")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = CLIConfig.from_cli()

    try:
        df = pd.read_pickle(cfg.input_file)
    except pickle.UnpicklingError:
        df = pd.read_csv(cfg.input_file)
    except Exception as exc:
        raise ValueError("Input must be a valid pickle or CSV") from exc

    print(f"Loaded {len(df):,} records from {cfg.input_file} …")

    # ---------------------------------------------------------------------
    # Optional AVS extraction branch – mirrors original flag behaviour
    # ---------------------------------------------------------------------
    if cfg.reproduce_avs_extraction:
        if "category" in df.columns:
            df = df[df["category"] == "Discharge summary"]

        df["brief_hospital_course"] = df["text"].apply(get_hospital_course)
        df["summary"] = df["text"].apply(get_instructions)

        good = lambda s: (s is not None) and len(s.split(" ")) >= 30
        df = df[df["summary"].apply(good) & df["brief_hospital_course"].apply(good)]

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        target = Path(cfg.output_dir) / "avs_mimic_processed_summaries.csv"
        df.to_csv(target, index=False)
        print(f"AVS‑style extraction written to {target}")
        return

    # ---------------------------------------------------------------------
    # Standard multi‑step pipeline – iterate over the registry
    # ---------------------------------------------------------------------
    for step_id in sorted(STEP_REGISTRY):
        if step_id < cfg.start_from_step:
            continue
        df = STEP_REGISTRY[step_id](df)
        print(f"    DataFrame now has {len(df):,} rows.")

	# Save
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(cfg.output_dir) / "mimic_processed_summaries.csv"
    df.to_csv(output_file, index=False)
    print(f"\nPipeline complete – results stored at {output_file}\n")


if __name__ == "__main__":
    main()
