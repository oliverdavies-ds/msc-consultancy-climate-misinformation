"""
Train a LoRA adapter that adapts the published CARDS classifier to UK discourse.

The LoRA configuration mirrors the one used to train the CARDS model itself.
Holding the training method constant means any change in accuracy can be
attributed to the added UK examples.

Run on a rented GPU. See docs/gpu_runs.md.

Input:  data/finetuning/uk_lora.jsonl
Output: finetuning/adapter/
"""

from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "finetuning" / "uk_lora.jsonl"
ADAPTER_DIR = ROOT / "finetuning" / "adapter"

BASE_MODEL = "C3DS/CARDS-Qwen3.6-27B"
MAX_SEQ_LENGTH = 4000
SEED = 42

# Loaded without 4 bit quantisation because the GPU had the memory for it.
# bfloat16 halves the footprint of full precision.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16, load_in_4bit=False, trust_remote_code=True)

# Rank 16 adapter on the attention and MLP projection layers, as in the original CARDS training.
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")


def to_text(example):
    """Render one chat example as a single training string using the model chat template."""
    # tokenize=False leaves tokenisation to the trainer. add_generation_prompt=False
    # because the assistant turn is already part of the example.
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,
                                                  add_generation_prompt=False)}


dataset = dataset.map(to_text, remove_columns=dataset.column_names)

# Effective batch size of 8 through gradient accumulation, 3 epochs, cosine schedule.
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=3, learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_steps=10, optim="adamw_8bit", bf16=True, logging_steps=5,
        max_seq_length=MAX_SEQ_LENGTH, dataset_text_field="text",
        output_dir="outputs", seed=SEED, report_to="none"))
trainer.train()

model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
