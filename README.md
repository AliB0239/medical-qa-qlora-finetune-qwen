# MedQA Fine-Tune

Fine-tuning **Qwen2.5-1.5B-Instruct** on the MedQuAD dataset using QLoRA,
as a structured learning exercise in LLM fine-tuning for medical question answering.

Built in two phases to demonstrate how fine-tuning can control both **domain knowledge**
and **output format** independently.

---

## What This Project Does

**Phase 1 — Domain Adaptation**
Fine-tune a general-purpose instruction model on medical Q&A pairs.
The goal is to make the model more knowledgeable and confident in the medical domain
while maintaining a natural, conversational tone.

**Phase 2 — Format Steering**
Retrain on the same data, but with answers reformatted into a consistent structure
(Overview / Details / When to See a Doctor / Disclaimer).
Same domain knowledge, different output shape.

This two-phase approach demonstrates that fine-tuning controls not just *what* a model
knows, but *how* it responds — a distinction that matters a lot in production AI systems.

---

## Dataset

[MedQuAD — Medical Question Answering Dataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)

- 16,407 medical Q&A pairs
- Sourced from NIH, NLM, and other trusted health organizations
- Question types: symptoms, diagnosis, treatment, susceptibility, and more
- Split: 90% train / 10% validation

---

## Model & Technique

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Technique** | QLoRA (4-bit quantization + LoRA adapters) |
| **LoRA rank** | 16 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate/up/down_proj |
| **Optimizer** | paged_adamw_8bit |
| **GPU used** | RTX 3050 4GB |
| **Effective batch size** | 8 (2 × 4 gradient accumulation steps) |

---

## Project Structure

```
medqa-finetune/
├── data/
│   ├── download.py        # fetch MedQuAD from HuggingFace
│   ├── prepare.py         # Phase 1: convert to chat format
│   └── reformat.py        # Phase 2: add structured sections
├── src/
│   ├── config.py          # all hyperparameters and paths
│   ├── model.py           # model loading, quantization, LoRA
│   ├── dataset.py         # data loading and tokenization
│   └── trainer.py         # SFTTrainer configuration
├── scripts/
│   ├── train.py           # training entry point
│   └── evaluate.py        # side-by-side model comparison
├── notebooks/
│   └── 01_exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# clone the repo
git clone https://github.com/your-username/medqa-finetune.git
cd medqa-finetune

# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Download the dataset
```bash
python data/download.py
```

### Step 2 — Prepare data (Phase 1)
```bash
python data/prepare.py
```

### Step 3 — Train (Phase 1)
```bash
python scripts/train.py --phase 1
```

### Step 4 — Evaluate Phase 1
```bash
python scripts/evaluate.py --phase 1
```

### Step 5 — Reformat data (Phase 2)
```bash
python data/reformat.py
```

### Step 6 — Train (Phase 2)
```bash
python scripts/train.py --phase 2
```

### Step 7 — Compare all three models
```bash
python scripts/evaluate.py --phase 2 --compare-all
```

---

## Example Output

**Question:** What are the early symptoms of Type 2 diabetes?

**Base Model (Qwen2.5-1.5B-Instruct, no fine-tuning)**
> Type 2 diabetes symptoms include increased thirst and frequent urination...
> *(general response, may lack specificity or completeness)*

**Phase 1 — Conversational**
> Early symptoms of Type 2 diabetes often develop gradually and can be subtle.
> Common signs include increased thirst (polydipsia), frequent urination (polyuria),
> unexplained fatigue, blurred vision, slow-healing wounds, and darkened skin in body
> creases. Many people with Type 2 diabetes have no symptoms initially, which is why
> regular screening is important, especially for those with risk factors.
> Please consult a healthcare professional for evaluation.

**Phase 2 — Structured**
> **Overview**
> Type 2 diabetes often develops gradually, and early symptoms can be easy to overlook.
>
> **Details**
> Common early signs include increased thirst, frequent urination, unexplained fatigue,
> blurred vision, and slow-healing sores. Some people also notice darkened skin patches
> around the neck or armpits (acanthosis nigricans).
>
> **When to See a Doctor**
> See your doctor if you notice these symptoms, especially with known risk factors
> like obesity, family history, or a sedentary lifestyle. Early diagnosis significantly
> improves outcomes.
>
> **Disclaimer**
> This information is for educational purposes only. Always consult a qualified
> healthcare professional for personal medical advice.

---

## Limitations

- This model is for **educational purposes only** — not for clinical use
- QLoRA fine-tuning improves domain knowledge but doesn't eliminate hallucination
- The structured output (Phase 2) relies on heuristic answer segmentation, which is imperfect
- Training on 4GB VRAM required conservative batch sizes and sequence lengths
- No RLHF or preference optimization — outputs are not safety-verified

---

## What I Learned

- The full QLoRA fine-tuning pipeline: quantization → LoRA → SFTTrainer → adapter saving
- How dataset formatting affects model output style independently of content
- Memory optimization techniques for training on consumer GPUs (paged optimizer, gradient checkpointing, fp16)
- The difference between domain adaptation and format steering as distinct fine-tuning objectives

---

## Acknowledgements

- Dataset: [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- Base model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Libraries: HuggingFace Transformers, PEFT, TRL, BitsAndBytes
