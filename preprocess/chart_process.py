from __future__ import annotations

import random, re, string
from typing import Callable, Dict
import pandas as pd
from preprocess.patterns import *


def get_instructions(text: str) -> str | None:
    start = text.find("Discharge Instructions:")
    end = text.find("Followup Instructions:")
    if start < 0 or end < 0:
        return None
    section = text[start:end].replace("\n", " ")
    return " ".join(section.split())


def get_hospital_course(text: str) -> str | None:
    start = text.find("Brief Hospital Course:")
    if start < 0:
        return None
    end_candidates = [
        text.find("Medications on Admission:"),
        text.find("Discharge Medications:"),
        text.find("Discharge Disposition:"),
    ]
    end_candidates = [p for p in end_candidates if p != -1]
    if not end_candidates:
        return None
    end = min(end_candidates)
    if end == 0 or start >= end:
        return None
    section = text[start:end].replace("\n", " ")
    section = " ".join(section.split())
    if len(section.split(" ")) < 30:  # quality gate – same as original
        return None
    return section


def strip_short_text(df: pd.DataFrame, min_len: int = 350) -> pd.DataFrame:
    before = len(df)
    df = df[df["summary"].str.len() > 0]
    df = df[df["summary"].str.len() >= min_len]
    removed = before - len(df)
    print(f"    Dropped {removed} summaries with < {min_len} chars or empty text.")
    return df


def why_waht_next_process(summaries: pd.Series) -> pd.Series:
    rand_mask = "".join(random.choices(string.ascii_uppercase + string.digits, k=20)) + "\n- "
    summaries = summaries.apply(lambda s: WHY_WHAT_NEXT_HEADINGS_DASHED_LIST.sub(rand_mask, s))

    dash_re = re.compile(r"(?:\.)?\n-\s{0,4}", re.MULTILINE | re.IGNORECASE)

    def _fix_paragraph(text: str) -> str:
        if rand_mask not in text:
            return text
        pieces = text.split(rand_mask)
        head, tails = pieces[0], pieces[1:]
        processed: list[str] = [head]
        for tail in tails:
            first_block, *rest = tail.split("\n\n", 1)
            first_block = ". ".join(dash_re.split(first_block))
            joined = first_block + ("\n\n" + rest[0] if rest else "")
            processed.append(joined.strip())
        return "\n\n".join(processed)

    return summaries.apply(_fix_paragraph)


def remove_regex_dict(
    df: pd.DataFrame,
    regexes: Dict[str, re.Pattern],
    postprocess: Callable[[str], str],
    keep: int = 0,
) -> pd.DataFrame:
    total_removed = 0

    # choose apply vs swifter.apply if swifter is present
    _apply = (
        lambda s, fn: s.swifter.apply(fn)
        if hasattr(pd.Series, "swifter")
        else s.apply(fn)
    )

    for label, pattern in regexes.items():
        mask = _apply(df["summary"], lambda text: bool(pattern.search(text)))
        n_changed = int(mask.sum())
        total_removed += n_changed

        if n_changed:
            df.loc[mask, "summary"] = _apply(
                df.loc[mask, "summary"],
                lambda text: postprocess(pattern.split(text, 1)[keep]),
            )

        print(f"  {label:<25} {n_changed:>6} / {len(df):,}")

    print(f"Changed a total of {total_removed:,} summaries.")
    return df
