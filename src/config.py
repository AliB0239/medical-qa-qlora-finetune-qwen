"""
config.py
---------
Single source of truth for all training hyperparameters, LoRA settings,
dataset parameters, and paths. Edit this file to switch between phases
or experiment with different settings.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"


# ─── Config Dataclasses ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_name: str = "models/qwen2.5-1.5b"
    load_in_4bit: bool = True           # QLoRA: quantize base model to 4-bit
    bnb_4bit_quant_type: str = "nf4"   # NormalFloat4 — best for QLoRA
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = False      # double quantization (saves ~0.4 GB, slightly slower)


@dataclass
class LoRAConfig:
    r: int = 16                         # rank — higher = more capacity, more VRAM
    lora_alpha: int = 32                # scaling factor (rule of thumb: 2x r)
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # which layers to apply LoRA to — attention projections are standard
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class DataConfig:
    dataset_id: str = "keivalya/MedQuad-MedicalQnADataset"
    local_raw_path: Path = DATA_DIR / "raw"
    local_prepared_path: Path = DATA_DIR / "prepared"
    local_reformatted_path: Path = DATA_DIR / "reformatted"   # phase 2
    train_split: float = 0.9
    val_split: float = 0.1
    max_seq_length: int = 512           # longer = more VRAM; 512 is safe for 4GB
    system_prompt: str = (
        "You are a knowledgeable and compassionate medical assistant. "
        "Answer medical questions clearly and accurately. "
        "Always remind users to consult a healthcare professional for personal medical advice."
    )


@dataclass
class TrainingConfig:
    # ── Phase selection ──────────────────────────────────────────────────────
    phase: int = 1                      # 1 = conversational, 2 = structured output

    # ── Core hyperparams ────────────────────────────────────────────────────
    num_train_epochs: int = 3
    # max_steps: int = 50
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch size = 2 * 4 = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # ── Memory optimization ──────────────────────────────────────────────────
    gradient_checkpointing: bool = True    # trades compute for VRAM savings
    optim: str = "paged_adamw_8bit"        # 8-bit optimizer — critical for 4GB
    fp16: bool = False                      # mixed precision training
    bf16: bool = True

    # ── Logging & saving ────────────────────────────────────────────────────
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2              # keep only last 2 checkpoints
    load_best_model_at_end: bool = True
    report_to: str = "none"               # change to "wandb" if you want W&B tracking

    @property
    def output_dir(self) -> Path:
        return OUTPUTS_DIR / f"phase{self.phase}"

    @property
    def logging_dir(self) -> Path:
        return OUTPUTS_DIR / f"phase{self.phase}" / "logs"


# ─── Convenience: instantiate defaults ────────────────────────────────────────

model_config = ModelConfig()
lora_config = LoRAConfig()
data_config = DataConfig()
training_config = TrainingConfig()
