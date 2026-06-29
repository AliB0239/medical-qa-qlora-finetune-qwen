"""
train.py
--------
Main training entry point. Ties together model loading, dataset loading,
and trainer setup, then runs training and saves the LoRA adapter.

Usage:
    # Phase 1 (conversational, default)
    python scripts/train.py

    # Phase 2 (structured output)
    python scripts/train.py --phase 2
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.config import training_config, model_config, lora_config
from src.model import load_base_model
from src.dataset import load_prepared_dataset, prepare_for_trainer
from src.trainer import build_trainer

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B on MedQuAD")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Training phase: 1 = conversational, 2 = structured output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # update phase in config
    training_config.phase = args.phase

    print("=" * 60)
    print(f"  MedQA Fine-Tuning — Phase {args.phase}")
    print("=" * 60)
    print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU   : {gpu.name}")
        print(f"VRAM  : {gpu.total_memory / 1e9:.1f} GB")

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    model, tokenizer = load_base_model(model_config, lora_config)

    # ── Step 2: Load dataset ──────────────────────────────────────────────────
    print("\n[2/4] Loading dataset...")
    train_ds, val_ds = load_prepared_dataset(phase=args.phase)
    train_ds, val_ds = prepare_for_trainer(train_ds, val_ds, tokenizer)

    # ── Step 3: Build trainer ─────────────────────────────────────────────────
    print("\n[3/4] Building trainer...")
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, training_config)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print("\n[4/4] Starting training...")
    print("      (this will take a while on a 4GB GPU — checkpoints saved every 100 steps)\n")

    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    adapter_path = training_config.output_dir / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\nTraining complete.")
    print(f"Adapter saved to: {adapter_path}")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/evaluate.py --phase {args.phase}")


if __name__ == "__main__":
    main()
