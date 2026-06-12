"""
Build preference data for the reward model and DPO from real public datasets:
  - Anthropic/hh-rlhf                       (helpful/harmless human preferences)
  - HuggingFaceH4/ultrafeedback_binarized   (LLM-judged preference pairs)

Emits JSONL of ``{"prompt", "preferred_response", "rejected_response"}`` by default
(train + held-out test). The held-out test split is what reward-model preference
accuracy is measured on. Use ``--legacy_labels`` to write the upstream
``{"prompt", "chosen", "rejected"}`` names instead.

Example:
    PYTHONPATH=. HF_HOME=/ephemeral/hf_cache python scripts/prepare_preference_data.py \
        --source both --max_per_source 40000 --out_dir /ephemeral/data
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")

_ASSISTANT_MARKER = "\n\nAssistant:"


def preference_row(prompt: str, preferred: str, rejected: str, *, legacy_labels: bool = False) -> dict:
    if legacy_labels:
        return {"prompt": prompt, "chosen": preferred, "rejected": rejected}
    return {
        "prompt": prompt,
        "preferred_response": preferred,
        "rejected_response": rejected,
    }


def _split_hh(text: str) -> tuple[str, str] | None:
    """Split an HH-RLHF conversation string into (prompt_context, final_response)."""
    idx = text.rfind(_ASSISTANT_MARKER)
    if idx == -1:
        return None
    prompt = text[:idx].strip()
    response = text[idx + len(_ASSISTANT_MARKER):].strip()
    if not prompt or not response:
        return None
    return prompt, response


def from_hh(max_n: int, split: str, *, legacy_labels: bool = False) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))
    out = []
    for ex in ds:
        c = _split_hh(ex["chosen"]); r = _split_hh(ex["rejected"])
        if not c or not r:
            continue
        prompt, chosen = c
        _, rejected = r
        if chosen == rejected:
            continue
        out.append(preference_row(prompt, chosen, rejected, legacy_labels=legacy_labels))
    return out


def from_ultrafeedback(max_n: int, split: str, *, legacy_labels: bool = False) -> list[dict]:
    from datasets import load_dataset

    hf_split = "train_prefs" if split == "train" else "test_prefs"
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=hf_split)
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))
    out = []
    for ex in ds:
        prompt = ex["prompt"].strip()
        chosen = ex["chosen"][-1]["content"].strip()
        rejected = ex["rejected"][-1]["content"].strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        out.append(preference_row(prompt, chosen, rejected, legacy_labels=legacy_labels))
    return out


def collect(source: str, max_n: int, split: str, *, legacy_labels: bool = False) -> list[dict]:
    rows: list[dict] = []
    if source in ("hh", "both"):
        print(f"Loading Anthropic/hh-rlhf [{split}] ...")
        rows += from_hh(max_n, split, legacy_labels=legacy_labels)
    if source in ("ultrafeedback", "both"):
        print(f"Loading ultrafeedback_binarized [{split}] ...")
        try:
            rows += from_ultrafeedback(max_n, split, legacy_labels=legacy_labels)
        except Exception as e:  # noqa: BLE001
            print(f"  (skipping ultrafeedback: {e})")
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} pairs -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["hh", "ultrafeedback", "both"], default="both")
    p.add_argument("--max_per_source", type=int, default=40000)
    p.add_argument("--out_dir", default="/ephemeral/data")
    p.add_argument("--legacy_labels", action="store_true", help="write chosen/rejected labels instead of preferred_response/rejected_response")
    args = p.parse_args()

    train = collect(args.source, args.max_per_source, "train", legacy_labels=args.legacy_labels)
    test = collect(args.source, max(2000, args.max_per_source // 20), "test", legacy_labels=args.legacy_labels)
    import random
    random.Random(0).shuffle(train)
    write_jsonl(train, os.path.join(args.out_dir, "preferences.jsonl"))
    write_jsonl(test, os.path.join(args.out_dir, "preferences_test.jsonl"))


if __name__ == "__main__":
    main()
