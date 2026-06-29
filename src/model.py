"""
model.py
--------
Handles loading the base model in 4-bit (QLoRA) and applying
the LoRA adapter via peft.

Two main entry points:
    load_base_model()   → quantized base model + tokenizer (for training)
    load_for_inference() → base model OR fine-tuned adapter (for evaluation)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from src.config import ModelConfig, LoRAConfig, model_config, lora_config


def get_bnb_config(cfg: ModelConfig) -> BitsAndBytesConfig:
    """Build the BitsAndBytes 4-bit quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg.use_nested_quant,
    )


def load_tokenizer(cfg: ModelConfig = model_config) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Qwen2.5 uses <|endoftext|> as both eos and pad — this is correct
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # right-pad for training (left-pad only needed for batch inference)
    tokenizer.padding_side = "right"

    return tokenizer


def load_base_model(
    model_cfg: ModelConfig = model_config,
    lora_cfg: LoRAConfig = lora_config,
) -> tuple:
    """
    Load the base model quantized to 4-bit and wrap it with LoRA adapters.
    Returns (model, tokenizer) ready for training.
    """
    print(f"Loading base model: {model_cfg.model_name}")
    print(f"  4-bit quantization : {model_cfg.load_in_4bit}")
    print(f"  LoRA rank          : {lora_cfg.r}")
    print(f"  Target modules     : {lora_cfg.target_modules}")

    tokenizer = load_tokenizer(model_cfg)

    bnb_config = get_bnb_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",          # puts model on GPU automatically
        trust_remote_code=True,
        local_files_only=True,
    )

    # required before applying LoRA to a quantized model
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # apply LoRA
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=lora_cfg.target_modules,
    )

    model = get_peft_model(model, peft_config)

    # print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params : {trainable:,} ({100 * trainable / total:.2f}% of total)")

    return model, tokenizer


def load_for_inference(
    adapter_path: str = None,
    model_cfg: ModelConfig = model_config,
) -> tuple:
    """
    Load a model for inference.
    - If adapter_path is None  → returns the raw base model (no LoRA)
    - If adapter_path is given → loads base model + merges the LoRA adapter

    Returns (model, tokenizer).
    """
    tokenizer = load_tokenizer(model_cfg)

    bnb_config = get_bnb_config(model_cfg)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    if adapter_path is None:
        print("Loaded: base model (no adapter)")
        return base_model, tokenizer

    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"Loaded: base model + adapter from {adapter_path}")
    return model, tokenizer
