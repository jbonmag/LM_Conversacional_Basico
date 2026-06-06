"""Fine-tune a small conversational language model with Hugging Face Transformers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test a small conversational language model.")
    parser.add_argument("--model-name", default="distilgpt2", help="Base causal language model.")
    parser.add_argument("--dataset-name", default="daily_dialog", help="Dataset available through Hugging Face Datasets.")
    parser.add_argument("--train-samples", type=int, default=1000, help="Maximum number of training examples.")
    parser.add_argument("--max-length", type=int, default=128, help="Token sequence length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--output-dir", default="trained_model", help="Directory to save the trained model.")
    parser.add_argument("--prompt", default=None, help="Optional prompt for a quick generation test after training.")
    return parser


def format_dialog(example: dict) -> dict:
    return {"text": " ".join(example["dialog"])}


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.map(lambda row: {"labels": row["input_ids"]}, batched=False)
    return tokenized


def train(args: argparse.Namespace) -> Path:
    os.environ.setdefault("WANDB_DISABLED", "true")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    raw_dataset = load_dataset(args.dataset_name, split=f"train[:{args.train_samples}]")
    text_dataset = raw_dataset.map(format_dialog, remove_columns=raw_dataset.column_names)
    tokenized_dataset = tokenize_dataset(text_dataset, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir="results",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def generate_sample(model_dir: Path, prompt: str) -> None:
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=str(model_dir), tokenizer=str(model_dir), device=device)
    result = generator(prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9)
    print(result[0]["generated_text"])


def main() -> None:
    args = build_parser().parse_args()
    model_dir = train(args)
    print(f"Model saved to: {model_dir}")

    if args.prompt:
        generate_sample(model_dir, args.prompt)


if __name__ == "__main__":
    main()
