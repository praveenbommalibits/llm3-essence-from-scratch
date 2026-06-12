# Data Labels and Training Meaning

This project has four data styles. Each one teaches a different behavior.

## 1. Pretraining Text

Schema:

```json
{"text": "Any raw document, article, code, or note."}
```

Used by:

- `scripts/data_preprocess.py`
- `scripts/prepare_pretrain_data.py`
- `data_loader/data_loader.py`

Meaning:

The model sees a stream of tokens and learns next-token prediction. There is no
human label such as positive/negative here. The next token is the training target.

## 2. SFT Instruction Labels

Canonical local schema:

```json
{"user": "Explain attention in one paragraph.", "assistant": "Attention lets each token compare itself with earlier tokens and mix useful context."}
```

Alternative schema:

```json
{"messages": [{"role": "user", "content": "Explain attention."}, {"role": "assistant", "content": "Attention is ..."}]}
```

Used by:

- `scripts/prepare_sft_data.py`
- `data_loader/sft_dataset.py`
- `src/post_training/chat_template.py`
- `scripts/train_sft.py`

Meaning:

The user text is context. The assistant text is the labeled target. During SFT,
`loss_mask` is `0` for prompt/user tokens and `1` for assistant tokens, so the
model is trained to produce the assistant answer rather than memorize the prompt.

## 3. Preference Labels

Canonical local schema:

```json
{"prompt": "Explain dropout.", "preferred_response": "Dropout randomly disables activations during training so the network cannot rely too heavily on one path.", "rejected_response": "Dropout means deleting the model."}
```

Upstream-compatible schema:

```json
{"prompt": "Explain dropout.", "chosen": "Clear and correct answer.", "rejected": "Unclear or wrong answer."}
```

Used by:

- `scripts/prepare_preference_data.py`
- `data_loader/preference_dataset.py`
- `scripts/train_reward.py`
- `scripts/train_dpo.py`

Meaning:

Both responses answer the same prompt. `preferred_response` is the better answer,
and `rejected_response` is the worse answer. The reward model learns a scalar
score where:

```text
score(preferred_response) > score(rejected_response)
```

DPO uses the same pairs to directly move the policy toward preferred responses.

## 4. RL / Verifier Prompts

Schema:

```json
{"prompt": "What is 12 + 8?", "gold": 20.0}
```

Used by:

- `scripts/prepare_rl_prompts.py`
- `data_loader/prompt_dataset.py`
- `scripts/train_ppo.py`
- `scripts/train_grpo.py`
- `scripts/eval_post_training.py`

Meaning:

`prompt` is sampled during rollout. `gold` is the verifiable answer used by the
reward function. The model is rewarded when its generated answer matches the
gold value, with a small bonus for the `<answer>...</answer>` format.

## Generate Local Samples

Run:

```bash
python scripts/create_llm3_sample_data.py --out_dir data/samples
```

This writes tiny JSONL files that demonstrate every schema:

- `pretrain_text.jsonl`
- `sft_custom.jsonl`
- `preferences_custom.jsonl`
- `rl_prompts_custom.jsonl`

These samples are intentionally small. They are for understanding and smoke
testing the data path, not for producing a high-quality model.
