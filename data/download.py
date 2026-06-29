"""
download.py
-----------
Fetches the MedQuAD dataset from HuggingFace Hub and saves it locally
as a parquet file. Run this once before any other data step.

Usage:
    python data/download.py
"""

import sys
from pathlib import Path

# make src importable from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.config import data_config


def download():
    print(f"Downloading dataset: {data_config.dataset_id}")

    dataset = load_dataset(data_config.dataset_id)

    # the dataset only has a 'train' split — we'll do our own split in prepare.py
    print(f"  Rows: {len(dataset['train'])}")
    print(f"  Columns: {dataset['train'].column_names}")
    print(f"  Sample row:")
    print(f"    qtype  : {dataset['train'][0]['qtype']}")
    print(f"    Question: {dataset['train'][0]['Question'][:80]}...")
    print(f"    Answer  : {dataset['train'][0]['Answer'][:80]}...")

    # save locally
    save_path = data_config.local_raw_path
    save_path.mkdir(parents=True, exist_ok=True)

    dataset["train"].to_parquet(save_path / "medquad.parquet")
    print(f"\nSaved to: {save_path / 'medquad.parquet'}")


if __name__ == "__main__":
    download()
