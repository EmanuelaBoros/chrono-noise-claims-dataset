#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upload ChronoNoise-Claims JSONL to Hugging Face Hub.

Example:

export HF_TOKEN="hf_..."

python upload_claims_to_hf.py \
  --jsonl outputs/chrononoise_claims_fr.jsonl \
  --hub-dataset-id EmanuelaBoros/chrononoise-claims-fr \
  --private
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import login


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def as_string(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def json_string(x: Any) -> str:
    if x is None:
        x = {}
    return json.dumps(x, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


# ---------------------------------------------------------------------
# Flatten records for stable HF schema
# ---------------------------------------------------------------------

def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": as_string(record.get("id")),
        "source_dataset": as_string(record.get("source_dataset")),
        "source_record_id": as_string(record.get("source_record_id")),
        "language": as_string(record.get("language")),
        "title": as_string(record.get("title")),
        "publication_date": as_string(record.get("publication_date")),
        "evidence_mode": as_string(record.get("evidence_mode")),
        "ocr_text": as_string(record.get("ocr_text")),
        "corrected_text": as_string(record.get("corrected_text")),
        "evidence_context": as_string(record.get("evidence_context")),
        "llm_output": as_string(record.get("llm_output")),
        "global_label": as_string(record.get("global_label")),
        "global_risk_summary": as_string(record.get("global_risk_summary")),
        "claims_json": json_string(record.get("claims")),
        "claim_verifications_json": json_string(record.get("claim_verifications")),
        "source_metadata_json": json_string(record.get("source_metadata")),
        "input_record_json": json_string(record.get("input_record")),
    }


def features() -> Features:
    return Features(
        {
            "id": Value("string"),
            "source_dataset": Value("string"),
            "source_record_id": Value("string"),
            "language": Value("string"),
            "title": Value("string"),
            "publication_date": Value("string"),
            "evidence_mode": Value("string"),
            "ocr_text": Value("string"),
            "corrected_text": Value("string"),
            "evidence_context": Value("string"),
            "llm_output": Value("string"),
            "global_label": Value("string"),
            "global_risk_summary": Value("string"),
            "claims_json": Value("string"),
            "claim_verifications_json": Value("string"),
            "source_metadata_json": Value("string"),
            "input_record_json": Value("string"),
        }
    )


def build_dataset_dict(
    jsonl_path: str,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    records = load_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    records = [flatten_record(r) for r in records]
    random.Random(seed).shuffle(records)

    n_total = len(records)
    if n_total < 10:
        n_val = 0
        n_test = 0
    else:
        n_val = int(n_total * validation_ratio)
        n_test = int(n_total * test_ratio)

    test_records = records[:n_test]
    val_records = records[n_test : n_test + n_val]
    train_records = records[n_test + n_val :]

    feats = features()

    ds = DatasetDict()
    ds["train"] = Dataset.from_list(train_records, features=feats)

    if val_records:
        ds["validation"] = Dataset.from_list(val_records, features=feats)

    if test_records:
        ds["test"] = Dataset.from_list(test_records, features=feats)

    return ds


# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------

def write_readme(out_dir: str, hub_dataset_id: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    readme = f"""---
language:
- fr
task_categories:
- text-classification
- text2text-generation
pretty_name: ChronoNoise Claims FR
tags:
- historical-newspapers
- ocr
- hallucination-detection
- claim-verification
- historical-nlp
- cultural-heritage
- llm-generated
---

# ChronoNoise-Claims-FR

ChronoNoise-Claims-FR is a silver dataset for evaluating whether LLM-generated historical claims are supported by noisy OCR text, post-corrected text, and document metadata.

The dataset studies the chain:

```text
historical OCR noise → LLM interpretation → evidence-grounded historical claims
```

## Main fields

- `ocr_text`: original OCR paragraph.
- `corrected_text`: post-corrected paragraph.
- `evidence_mode`: whether the model saw OCR, corrected text, or both.
- `llm_output`: generated historical summary.
- `claims_json`: JSON string containing atomic claims.
- `claim_verifications_json`: JSON string containing support labels, evidence spans, risk types, and explanations.
- `global_label`: overall support label for the generated output.
- `global_risk_summary`: short description of the main risk.

## Claim labels

Claim-level support labels include:

- `SUPPORTED`
- `INFERRED_SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `CONTRADICTED`
- `UNCERTAIN_DUE_TO_OCR`
- `TEMPORALLY_INVALID`
- `ENTITY_DRIFT`
- `OVER_SPECIFIED`

## Loading JSON columns

```python
import json
from datasets import load_dataset

ds = load_dataset("{hub_dataset_id}")
ex = ds["train"][0]

claims = json.loads(ex["claims_json"])
verifications = json.loads(ex["claim_verifications_json"])
```

## Note

This is a silver dataset. Claims and verification labels are automatically generated and should be manually verified before use as gold-standard data.
"""

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload ChronoNoise-Claims JSONL to HF Hub.")

    parser.add_argument("--jsonl", required=True, help="Path to claims JSONL file.")
    parser.add_argument("--hub-dataset-id", required=True, help="Example: EmanuelaBoros/chrononoise-claims-fr")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Defaults to HF_TOKEN env var.")
    parser.add_argument("--private", action="store_true", help="Push as private dataset.")
    parser.add_argument("--out-dir", default="hf_dataset_claims", help="Local HF dataset folder.")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--commit-message", default="Upload ChronoNoise-Claims dataset")
    parser.add_argument("--no-push", action="store_true", help="Only save locally, do not push.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[load] JSONL: {args.jsonl}")
    ds = build_dataset_dict(
        jsonl_path=args.jsonl,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(ds)
    print("[features]")
    print(ds["train"].features)

    print(f"[save] Saving locally to: {args.out_dir}")
    ds.save_to_disk(args.out_dir)
    write_readme(args.out_dir, args.hub_dataset_id)

    if args.no_push:
        print("[done] Saved locally only. Not pushed because --no-push was used.")
        return

    if args.hf_token:
        print("[hf] Logging in")
        login(token=args.hf_token)
    else:
        print("[hf] No token provided; assuming you already ran huggingface-cli login.")

    print(f"[push] Uploading to: {args.hub_dataset_id}")
    ds.push_to_hub(
        args.hub_dataset_id,
        private=args.private,
        commit_message=args.commit_message,
    )

    print(f"[done] https://huggingface.co/datasets/{args.hub_dataset_id}")


if __name__ == "__main__":
    main()
