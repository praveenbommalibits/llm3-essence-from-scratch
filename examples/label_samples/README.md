# Label Samples

These tiny files mirror the schemas described in `docs/DATA_LABELS.md`.

- `pretrain_text.jsonl`: raw text for next-token pretraining
- `sft_custom.jsonl`: user prompts with assistant targets
- `preferences_custom.jsonl`: preferred vs rejected response labels
- `rl_prompts_custom.jsonl`: prompts with numeric gold answers

They are intentionally tiny, so they are useful for reading and smoke testing,
not for serious training.
