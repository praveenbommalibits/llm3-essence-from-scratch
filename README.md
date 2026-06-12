# LLM3 Essence From Scratch

An educational PyTorch project for understanding the full path from a tiny
decoder-only transformer to post-training:

1. pretraining on next-token prediction data
2. supervised fine-tuning (SFT) with prompt-masked assistant responses
3. reward modeling with preference pairs
4. DPO / PPO / GRPO style alignment experiments
5. simple inference, evaluation, docs, and a Streamlit control panel

This repository is an adapted learning workspace based on Fareed Khan's MIT
licensed [`train-llm-from-scratch`](https://github.com/FareedKhan-dev/train-llm-from-scratch)
project. The original license is preserved in `LICENSE`, and the upstream README
is kept at `docs/UPSTREAM_README.md`.

## What I Changed

- Reframed the project as **LLM3 Essence From Scratch** for hands-on learning.
- Added a data-label guide at `docs/DATA_LABELS.md`.
- Added `scripts/create_llm3_sample_data.py` to generate small local sample files
  for pretraining, SFT, preference training, and RL prompts.
- Added checked-in sample label files under `examples/label_samples/`.
- Made preference data labels easier to read:
  - canonical local schema: `prompt`, `preferred_response`, `rejected_response`
  - upstream compatibility: `prompt`, `chosen`, `rejected`
- Added smoke-test coverage for the clearer preference label schema.

## Code Map

```text
src/models/                 Transformer, attention, MLP, and blocks
src/post_training/          SFT, reward model, DPO, PPO, GRPO, rollout, eval
data_loader/                HDF5, SFT, preference, and RL prompt iterators
scripts/                    Data prep, training, evaluation, chat, samples
configs/                    JSON configs for each training stage
ui/                         Streamlit training/evaluation/chat control panel
docs/                       Theory notes, diagrams, and label documentation
tests/                      CPU smoke tests for the core math and data flow
```

## Label Schemas

The most important data formats are:

```json
{"text": "Raw document text for next-token pretraining."}
```

```json
{"user": "Explain attention simply.", "assistant": "Attention lets each token compare itself with earlier tokens."}
```

```json
{"prompt": "Explain dropout.", "preferred_response": "A clear, correct explanation.", "rejected_response": "A vague or wrong answer."}
```

```json
{"prompt": "What is 12 + 8?", "gold": 20.0}
```

See `docs/DATA_LABELS.md` for how each label is used during training.

## Quick Start

Create the local example data:

```bash
python scripts/create_llm3_sample_data.py --out_dir data/samples
```

Or inspect the checked-in examples under `examples/label_samples/`.

Run the lightweight smoke tests:

```bash
PYTHONPATH=. python tests/test_post_training_smoke.py
PYTHONPATH=. python tests/test_rl_math.py
```

Prepare custom SFT data from the sample file only:

```bash
PYTHONPATH=. python scripts/prepare_sft_data.py \
  --custom_sft_jsonl data/samples/sft_custom.jsonl \
  --skip_public \
  --context_length 64 \
  --out_dir data/samples/packed
```

Train or evaluate with the existing stage scripts after choosing the config size
that matches your machine. The smoke configs in `configs/smoke/` are the safest
starting point for CPU or small-GPU experiments.

## Learning Notes

At a high level, the model learns in stages:

- **Pretraining:** read raw `text`, tokenize it, and predict the next token.
- **SFT:** read user/assistant examples, but compute loss only on assistant tokens.
- **Reward model:** compare two answers for the same prompt and learn which one is
  preferred.
- **DPO/PPO/GRPO:** use preference or verifiable reward signals to make responses
  more useful and better formatted.

This project is meant for study and portfolio explanation. When sharing it, keep
the upstream attribution and describe your own changes honestly.
