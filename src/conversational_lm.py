"""Fine-tune a small causal language model (distilgpt2) on DailyDialog and chat with it."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# DailyDialog (Li et al., 2017) in Parquet format. The original "daily_dialog" repository
# relies on a loading script, which datasets >= 4.0 no longer runs.
DATASET = "OpenRL/daily_dialog"
DATASET_REVISION = "1668faf0c0dc44664f108c489fd0666128db2c48"
BASE_MODEL = "distilgpt2"
SPEAKERS = ("A", "B")
SEED = 42


def clean_turn(text: str) -> str:
    """DailyDialog is tokenised with spaces around punctuation ("it ' s good ."); undo that."""
    text = text.replace("’", "'").strip()
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"\s*'\s*(s|t|re|ve|ll|d|m)\b", r"'\1", text)
    return re.sub(r"\s{2,}", " ", text)


def format_dialog(turns: list[str], eos: str) -> str:
    """One turn per line, speakers alternating A/B, end of the dialogue marked with EOS."""
    lines = [f"{SPEAKERS[i % 2]}: {clean_turn(t)}" for i, t in enumerate(turns) if t.strip()]
    return "\n".join(lines) + eos


def load_splits(train_samples: int | None, eval_samples: int):
    data = load_dataset(DATASET, revision=DATASET_REVISION)
    train = data["train"].shuffle(seed=SEED)
    if train_samples:
        train = train.select(range(min(train_samples, len(train))))
    validation = data["validation"].select(range(min(eval_samples, len(data["validation"]))))
    return train, validation


def tokenize_split(split, tokenizer, max_length: int):
    def encode(batch):
        texts = [format_dialog(d, tokenizer.eos_token) for d in batch["dialog"]]
        return tokenizer(texts, truncation=True, max_length=max_length)

    return split.map(encode, batched=True, remove_columns=split.column_names)


class CausalLMCollator:
    """Pads a batch and masks only the padding in the labels.

    GPT-2 has no pad token, so EOS doubles as padding. The stock DataCollatorForLanguageModeling
    masks every pad-token id in the labels, which would also hide the real end-of-dialogue EOS,
    and the model would never learn to stop. Masking by attention mask keeps it.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def load_model(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def build_trainer(model, tokenizer, train_ds, eval_ds, args) -> Trainer:
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalLMCollator(tokenizer),
    )


def perplexity(trainer: Trainer) -> float:
    return math.exp(trainer.evaluate()["eval_loss"])


@torch.no_grad()
def reply(model, tokenizer, history: list[str], max_new_tokens: int = 40, sample: bool = True) -> str:
    """Continue a conversation: `history` alternates A/B starting with A; returns the next turn."""
    lines = [f"{SPEAKERS[i % 2]}: {turn}" for i, turn in enumerate(history)]
    prompt = "\n".join(lines) + f"\n{SPEAKERS[len(history) % 2]}:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    newline_id = tokenizer.encode("\n")[0]
    sampling = {"do_sample": True, "top_p": 0.9, "temperature": 0.7} if sample else {"do_sample": False}
    output = model.generate(
        **inputs,
        **sampling,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
        eos_token_id=[newline_id, tokenizer.eos_token_id],
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0]


def train(args) -> None:
    set_seed(SEED)
    tokenizer, model = load_model(args.model_name)
    train_raw, eval_raw = load_splits(args.train_samples, args.eval_samples)
    train_ds = tokenize_split(train_raw, tokenizer, args.max_length)
    eval_ds = tokenize_split(eval_raw, tokenizer, args.max_length)
    print(f"Training dialogues: {len(train_ds)}  validation dialogues: {len(eval_ds)}")

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, args)
    before = perplexity(trainer)
    trainer.train()
    after = perplexity(trainer)
    print(f"Validation perplexity: {before:.1f} before fine-tuning, {after:.1f} after")

    output_dir = Path(args.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def chat(args) -> None:
    set_seed(SEED)
    tokenizer, model = load_model(args.model_dir)
    model.eval()
    history: list[str] = []
    print("Type a message (empty line to quit).")
    while True:
        message = input("you> ").strip()
        if not message:
            break
        history.append(message)
        answer = reply(model, tokenizer, history[-5:])  # odd length, so the user is always A
        history.append(answer)
        print(f"bot> {answer}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Fine-tune the base model on DailyDialog.")
    t.add_argument("--model-name", default=BASE_MODEL)
    t.add_argument("--train-samples", type=int, default=2000, help="Dialogues to train on (0 = all 11,118).")
    t.add_argument("--eval-samples", type=int, default=500, help="Validation dialogues for perplexity.")
    t.add_argument("--max-length", type=int, default=256)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--learning-rate", type=float, default=5e-5)
    t.add_argument("--output-dir", default="trained_model")
    t.set_defaults(func=train)

    c = sub.add_parser("chat", help="Talk to a fine-tuned model in the terminal.")
    c.add_argument("--model-dir", default="trained_model")
    c.set_defaults(func=chat)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
