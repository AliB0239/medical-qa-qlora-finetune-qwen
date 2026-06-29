"""
dataset.py
----------
Loads the prepared parquet files and tokenizes them using the model's
built-in chat template (apply_chat_template).

The SFTTrainer from trl expects a dataset with a "messages" column
in the standard chat format — which is exactly what prepare.py produces.
So this module is intentionally lightweight; heavy lifting is in prepare.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from transformers import AutoTokenizer
from src.config import data_config, DataConfig


def load_prepared_dataset(phase: int = 1, cfg: DataConfig = data_config) -> tuple[Dataset, Dataset]:
    """
    Load train and validation splits from disk.

    Args:
        phase: 1 = conversational (prepare.py output)
               2 = structured (reformat.py output)

    Returns:
        (train_dataset, val_dataset)
    """
    if phase == 1:
        data_path = cfg.local_prepared_path
    elif phase == 2:
        data_path = cfg.local_reformatted_path
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")

    train_path = data_path / "train.parquet"
    val_path = data_path / "validation.parquet"

    if not train_path.exists() or not val_path.exists():
        script = "data/prepare.py" if phase == 1 else "data/reformat.py"
        raise FileNotFoundError(
            f"Prepared data not found at {data_path}. "
            f"Run `python {script}` first."
        )

    train_ds = Dataset.from_parquet(str(train_path))
    val_ds = Dataset.from_parquet(str(val_path))

    print(f"Loaded Phase {phase} dataset:")
    print(f"  Train rows      : {len(train_ds)}")
    print(f"  Validation rows : {len(val_ds)}")
    print(f"  Columns         : {train_ds.column_names}")

    return train_ds, val_ds


def format_chat(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Apply the model's chat template to a single example.
    Used as a map() function during tokenization.

    The SFTTrainer can also handle this automatically if you pass
    the dataset with a 'messages' column and set the tokenizer —
    but doing it explicitly gives you more control and visibility.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def prepare_for_trainer(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: AutoTokenizer,
) -> tuple[Dataset, Dataset]:
    """
    Apply chat template to train and val sets.
    Returns datasets with a 'text' column, ready for SFTTrainer.
    """
    train_ds = train_ds.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=train_ds.column_names,
        desc="Formatting train set",
    )

    val_ds = val_ds.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=val_ds.column_names,
        desc="Formatting validation set",
    )

    # quick sanity check
    print("\nSample formatted text (first 300 chars):")
    print(train_ds[0]["text"][:300])
    print("...")

    return train_ds, val_ds
