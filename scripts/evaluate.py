"""
evaluate.py
-----------
Runs a set of sample medical questions through both the base model
and the fine-tuned adapter, printing outputs side by side.

This is the "show your work" script — the output goes into your README.

Usage:
    # Compare base vs phase 1
    python scripts/evaluate.py --phase 1

    # Compare base vs phase 1 vs phase 2 (run after both phases are trained)
    python scripts/evaluate.py --phase 2 --compare-all
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import pipeline
from src.config import training_config, model_config, data_config


# ── Sample questions ──────────────────────────────────────────────────────────
# These should NOT be in the training set — they're for qualitative evaluation.

EVAL_QUESTIONS = [
    "What are the early symptoms of Type 2 diabetes?",
    "How is high blood pressure diagnosed?",
    "What causes migraines and how can they be prevented?",
    "What is the difference between a cold and the flu?",
    "What are the treatment options for anxiety disorders?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MedQA model")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare base, phase 1, and phase 2 outputs (requires both phases trained)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Max tokens to generate per response",
    )
    return parser.parse_args()


def generate_response(pipe, question: str, max_new_tokens: int) -> str:
    """Generate a response for a single question using a pipeline."""
    messages = [
        {"role": "system", "content": data_config.system_prompt},
        {"role": "user", "content": question},
    ]
    output = pipe(messages, max_new_tokens=max_new_tokens)
    # extract just the assistant's reply
    return output[0]["generated_text"][-1]["content"].strip()


def load_pipeline(adapter_path: str = None) -> pipeline:
    """Load a text-generation pipeline, optionally with a LoRA adapter."""
    from src.model import load_for_inference

    model, tokenizer = load_for_inference(adapter_path=adapter_path)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    )


def print_comparison(question: str, responses: dict):
    """Pretty-print a question with all model responses."""
    separator = "─" * 70
    print(f"\n{separator}")
    print(f"QUESTION: {question}")
    print(separator)

    for label, response in responses.items():
        print(f"\n[{label}]")
        print(response)

    print()


def main():
    args = parse_args()
    training_config.phase = args.phase

    adapter_paths = {}

    # always include base model
    adapter_paths["Base Model"] = None

    # phase 1 adapter
    p1_adapter = training_config.output_dir.parent / "phase1" / "final_adapter"
    if p1_adapter.exists():
        adapter_paths["Phase 1 (Conversational)"] = str(p1_adapter)
    elif args.phase == 1:
        print(f"Phase 1 adapter not found at {p1_adapter}")
        print("Run `python scripts/train.py --phase 1` first.")
        sys.exit(1)

    # phase 2 adapter (only if --compare-all)
    if args.compare_all:
        p2_adapter = training_config.output_dir.parent / "phase2" / "final_adapter"
        if p2_adapter.exists():
            adapter_paths["Phase 2 (Structured)"] = str(p2_adapter)
        else:
            print(f"Phase 2 adapter not found at {p2_adapter}. Skipping.")

    print("=" * 70)
    print("  MedQA Evaluation")
    print("=" * 70)
    print(f"\nModels to compare: {list(adapter_paths.keys())}")
    print(f"Questions         : {len(EVAL_QUESTIONS)}")
    print(f"Max new tokens    : {args.max_new_tokens}")

    # load all pipelines
    pipelines = {}
    for label, adapter_path in adapter_paths.items():
        print(f"\nLoading: {label}...")
        pipelines[label] = load_pipeline(adapter_path)

    # run evaluation
    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)

    for question in EVAL_QUESTIONS:
        responses = {}
        for label, pipe in pipelines.items():
            responses[label] = generate_response(pipe, question, args.max_new_tokens)

        print_comparison(question, responses)


if __name__ == "__main__":
    main()
