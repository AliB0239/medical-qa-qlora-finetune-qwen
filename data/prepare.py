"""
prepare.py
----------
Phase 1 data preparation.

Converts raw MedQuAD rows into chat-formatted instruction pairs:
    system prompt + user question + assistant answer

Then splits into train/validation sets and saves as parquet.

Usage:
    python data/prepare.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import data_config


def row_to_chat(row: dict) -> dict:
    """Convert a single MedQuAD row into a chat-formatted messages list."""
    return {
        "messages": [
            {
                "role": "system",
                "content": data_config.system_prompt,
            },
            {
                "role": "user",
                "content": row["Question"].strip(),
            },
            {
                "role": "assistant",
                "content": row["Answer"].strip(),
            },
        ]
    }


def filter_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or very short questions/answers."""
    before = len(df)

    df = df.dropna(subset=["Question", "Answer"])
    df = df[df["Question"].str.strip().str.len() > 10]
    df = df[df["Answer"].str.strip().str.len() > 20]

    after = len(df)
    print(f"  Filtered {before - after} bad rows ({before} → {after})")
    return df


def prepare():
    raw_path = data_config.local_raw_path / "medquad.parquet"

    if not raw_path.exists():
        print("Raw data not found. Run `python data/download.py` first.")
        sys.exit(1)

    print("Loading raw data...")
    df = pd.read_parquet(raw_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Question types: {df['qtype'].value_counts().to_dict()}")

    print("\nCleaning...")
    df = filter_bad_rows(df)

    print("\nConverting to chat format...")
    records = [row_to_chat(row) for _, row in df.iterrows()]
    dataset = Dataset.from_list(records)

    print(f"\nSplitting {data_config.train_split:.0%} train / {data_config.val_split:.0%} val...")
    split = dataset.train_test_split(
        test_size=data_config.val_split,
        seed=42,
        shuffle=True,
    )

    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    print(f"  Train rows : {len(dataset_dict['train'])}")
    print(f"  Val rows   : {len(dataset_dict['validation'])}")

    # print a sample to verify format
    print("\nSample (train[0]):")
    for msg in dataset_dict["train"][0]["messages"]:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    save_path = data_config.local_prepared_path
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dict["train"].to_parquet(save_path / "train.parquet")
    dataset_dict["validation"].to_parquet(save_path / "validation.parquet")

    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    prepare()
