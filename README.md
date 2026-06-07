# Basic Conversational Language Model

Basic fine-tuning project for a conversational language model using Hugging Face Transformers. The example starts from `distilgpt2` and the public `daily_dialog` dataset to train a small causal language model for simple conversational responses.

## Objective

Provide a minimal and reproducible workflow to:

- Load a base language model.
- Prepare dialogues as training text.
- Tokenize examples with a fixed sequence length.
- Train with `Trainer`.
- Save the resulting model.
- Optionally run a quick text generation test.

## Structure

```text
.
|-- LM_Conversacional_basico_FIX.ipynb
|-- lm_conversacional_basico.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Installation

A virtual environment is recommended:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Basic training:

```bash
python lm_conversacional_basico.py
```

Quick training with fewer examples:

```bash
python lm_conversacional_basico.py --train-samples 100 --epochs 1
```

Training plus a generation test:

```bash
python lm_conversacional_basico.py --prompt "Hello, how are you?"
```

## Main Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-name` | Base causal language model | `distilgpt2` |
| `--dataset-name` | Training dataset | `daily_dialog` |
| `--train-samples` | Maximum number of dialogues | `1000` |
| `--max-length` | Maximum token length | `128` |
| `--epochs` | Training epochs | `1` |
| `--batch-size` | Per-device batch size | `2` |
| `--output-dir` | Model output directory | `trained_model` |

## Notes

The trained model and checkpoints are not versioned because they can become large. To preserve a result, store the output directory outside the repository or publish it to a model registry.

## Security

The default setup uses public resources and does not require tokens. Do not store credentials in notebooks, scripts or execution outputs.
