"""
reformat.py
-----------
Phase 2 data preparation.

Takes the same raw MedQuAD data and reformats each answer into a
structured template with clear sections. The model trained on this
will produce consistently formatted responses — useful for downstream
parsing or display in a UI.

Structured answer template:
    **Overview**
    <brief summary>

    **Details**
    <main explanation>

    **When to See a Doctor**
    <relevant guidance>

    **Disclaimer**
    This information is for educational purposes only...

Usage:
    python data/reformat.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import data_config


SYSTEM_PROMPT_PHASE2 = (
    "You are a knowledgeable and compassionate medical assistant. "
    "Always respond in the following structured format:\n\n"
    "**Overview**\n<brief 1-2 sentence summary>\n\n"
    "**Details**\n<main explanation>\n\n"
    "**When to See a Doctor**\n<relevant guidance>\n\n"
    "**Disclaimer**\n"
    "This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for personal medical advice."
)


def extract_sections(answer: str) -> dict:
    """
    Heuristically split a raw MedQuAD answer into sections.
    MedQuAD answers are typically a few paragraphs — we map them to
    our structured template by sentence position.
    """
    # clean up whitespace
    answer = re.sub(r'\n{3,}', '\n\n', answer.strip())
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]

    if not sentences:
        return None

    # overview: first 1-2 sentences
    overview_end = min(2, len(sentences))
    overview = " ".join(sentences[:overview_end])

    # details: middle sentences
    details_sentences = sentences[overview_end:]
    if not details_sentences:
        details_sentences = sentences  # fallback: repeat all

    # when to see a doctor: look for keywords, else use last sentence
    doctor_keywords = ["doctor", "physician", "medical", "seek", "consult", "emergency", "hospital", "clinic"]
    doctor_sentences = [
        s for s in details_sentences
        if any(kw in s.lower() for kw in doctor_keywords)
    ]

    if doctor_sentences:
        when_to_see = " ".join(doctor_sentences)
        # remove from details
        details = " ".join([s for s in details_sentences if s not in doctor_sentences])
    else:
        when_to_see = "Consult a healthcare professional if symptoms persist or worsen."
        details = " ".join(details_sentences)

    if not details:
        details = overview

    return {
        "overview": overview,
        "details": details,
        "when_to_see": when_to_see,
    }


def format_structured_answer(sections: dict) -> str:
    """Assemble sections into the structured markdown response."""
    return (
        f"**Overview**\n{sections['overview']}\n\n"
        f"**Details**\n{sections['details']}\n\n"
        f"**When to See a Doctor**\n{sections['when_to_see']}\n\n"
        f"**Disclaimer**\n"
        f"This information is for educational purposes only. "
        f"Always consult a qualified healthcare professional for personal medical advice."
    )


def row_to_structured_chat(row: dict) -> dict | None:
    """Convert a MedQuAD row into a structured chat format."""
    sections = extract_sections(row["Answer"])
    if sections is None:
        return None

    structured_answer = format_structured_answer(sections)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_PHASE2},
            {"role": "user", "content": row["Question"].strip()},
            {"role": "assistant", "content": structured_answer},
        ]
    }


def reformat():
    raw_path = data_config.local_raw_path / "medquad.parquet"

    if not raw_path.exists():
        print("Raw data not found. Run `python data/download.py` first.")
        sys.exit(1)

    print("Loading raw data...")
    df = pd.read_parquet(raw_path)
    df = df.dropna(subset=["Question", "Answer"])
    print(f"  Rows: {len(df)}")

    print("\nReformatting answers into structured format...")
    records = []
    skipped = 0
    for _, row in df.iterrows():
        result = row_to_structured_chat(row)
        if result:
            records.append(result)
        else:
            skipped += 1

    print(f"  Converted : {len(records)}")
    print(f"  Skipped   : {skipped}")

    # show a sample
    print("\nSample structured answer:")
    print(records[0]["messages"][2]["content"][:400])
    print("...")

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=data_config.val_split, seed=42, shuffle=True)

    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    save_path = data_config.local_reformatted_path
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dict["train"].to_parquet(save_path / "train.parquet")
    dataset_dict["validation"].to_parquet(save_path / "validation.parquet")

    print(f"\nSaved to: {save_path}")
    print(f"  Train : {len(dataset_dict['train'])} rows")
    print(f"  Val   : {len(dataset_dict['validation'])} rows")
    print(f"\nNext step: python scripts/train.py --phase 2")


if __name__ == "__main__":
    reformat()
