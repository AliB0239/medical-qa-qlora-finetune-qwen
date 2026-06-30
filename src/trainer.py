"""
trainer.py
----------
Configures and returns the SFTTrainer from trl.

SFTTrainer (Supervised Fine-Tuning Trainer) is a thin wrapper around
HuggingFace Trainer that handles:
  - packing sequences for efficiency
  - applying chat templates automatically (we do this manually instead)
  - integration with peft LoRA adapters
"""

from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from peft import PeftModel

from src.config import TrainingConfig, DataConfig, training_config, data_config


def build_training_args(train_cfg: TrainingConfig = training_config) -> SFTConfig:
    """
    Build the SFTConfig (extends TrainingArguments with SFT-specific settings).
    """
    return SFTConfig(
        # ── output & logging ────────────────────────────────────────────────
        output_dir=str(train_cfg.output_dir),
        logging_dir=str(train_cfg.logging_dir),
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,

        # ── training duration ────────────────────────────────────────────────
        num_train_epochs=train_cfg.num_train_epochs,
        #max_steps=train_cfg.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,

        # ── optimizer & scheduler ────────────────────────────────────────────
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        optim=train_cfg.optim,

        # ── memory optimization ──────────────────────────────────────────────
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,

        # ── evaluation & checkpointing ───────────────────────────────────────
        eval_strategy="steps",
        eval_steps=train_cfg.eval_steps,
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=True,

        # ── SFT-specific ─────────────────────────────────────────────────────
        max_length=data_config.max_seq_length,
        dataset_text_field="text",      # column name in our prepared dataset
        packing=False,                  # disable sequence packing (simpler, fine for our size)
    )


def build_trainer(
    model,
    tokenizer,
    train_ds: Dataset,
    val_ds: Dataset,
    train_cfg: TrainingConfig = training_config,
) -> SFTTrainer:
    """
    Assemble and return the SFTTrainer.
    """
    training_args = build_training_args(train_cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"Trainer ready.")
    print(f"  Phase          : {train_cfg.phase}")
    print(f"  Output dir     : {train_cfg.output_dir}")
    print(f"  Epochs         : {train_cfg.num_train_epochs}")
    print(f"  Effective batch: {train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps}")
    print(f"  Learning rate  : {train_cfg.learning_rate}")

    return trainer
