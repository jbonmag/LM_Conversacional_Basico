# LM_Conversacional_Basico

Fine-tunes a small language model, [`distilgpt2`](https://huggingface.co/distilbert/distilgpt2) (82M parameters), on [DailyDialog](https://aclanthology.org/I17-1099/), a set of everyday English conversations, so that it answers turn by turn.

| | validation perplexity |
|---|---|
| `distilgpt2` as downloaded | 20.6 |
| after fine-tuning on 2,000 dialogues, 1 epoch | **13.8** (33% lower) |

Perplexity is measured on the first 500 validation dialogues, which are never used for training. Training took 19.5 minutes on a CPU (14 threads); a GPU is picked up automatically if there is one.

## What it does and does not learn

The same prompts, answered with greedy decoding before and after fine-tuning (from the executed notebook):

| prompt (A) | `distilgpt2` (B) | fine-tuned (B) |
|---|---|---|
| What would you like for dinner tonight? | I'd love to have a good meal. But it's not going to be easy, because we're all in the same boat and there are so many different things that can happen at any time! | I'd love to have a good meal. |
| I'm thinking about changing jobs. | What do you think of the new job market? | What's the problem? Is it a job loss or an accident, and how do you know that your company is in trouble for hiring people who are not qualified to work full time as employees of our |

After fine-tuning the model has picked up the format (it answers as the other speaker, one turn at a time) and the register of everyday conversation, which is what the lower perplexity measures. It does not understand what it is told: many replies still drift off topic or run on until the length limit. That is the expected ceiling for an 82M-parameter model and a few thousand dialogues, not a bug. The notebook shows all the prompts and a short multi-turn conversation.

## How it works

1. **Data.** DailyDialog is loaded from a Parquet copy ([`OpenRL/daily_dialog`](https://huggingface.co/datasets/OpenRL/daily_dialog)) pinned to a fixed revision. The original `daily_dialog` repository depends on a loading script, which `datasets` 4.0 and later refuse to run.
2. **Formatting.** The spacing around punctuation is cleaned up (`it ' s good .` → `it's good.`). Each conversation becomes one turn per line, `A:` and `B:` alternating, followed by the end-of-text token.
3. **Training.** Causal language modelling with the Hugging Face `Trainer`. GPT-2 has no padding token, so the end-of-text token doubles as padding. A small custom collator masks the padding by attention mask, so the real end-of-text token stays in the loss and the model learns where a conversation ends. The stock `DataCollatorForLanguageModeling` would mask it too.
4. **Replying.** The conversation so far is written in the same format, ending with the next speaker's tag, and generation stops at the first newline.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/conversational_lm.py train          # about 20 min on CPU; writes trained_model/
python src/conversational_lm.py chat           # talk to it in the terminal
```

`train` accepts `--train-samples` (0 = all 11,118 dialogues), `--epochs`, `--batch-size`, `--learning-rate`, `--max-length` and `--output-dir`. More data and more epochs lower the perplexity further, at a proportional cost in time.

[`conversational_lm.ipynb`](conversational_lm.ipynb) runs the whole experiment (data, base model, training, evaluation, examples) and is saved with its outputs, so it can be read on GitHub without running anything.

## Files

```text
src/conversational_lm.py   data loading, formatting, collator, training, evaluation, chat
conversational_lm.ipynb    the experiment, executed
requirements.txt
```

The trained model is not versioned (`trained_model/` is in `.gitignore`).

## License

The code is under the [MIT license](LICENSE). The data is not: DailyDialog is released under CC BY-NC-SA 4.0, so models trained on it are for non-commercial use.
