#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ChronoNoise-Claims from a ChronoCorrect-Europeana-style JSONL file.

ChronoNoise-Claims evaluates whether LLM-generated historical claims are
supported by noisy OCR text, corrected text, and document metadata.

Typical generation:

export OPENAI_API_KEY="sk-..."

python build_claims_dataset.py \
  --input-jsonl ../europeana-post-correct-data/outputs/chronocorrect_europeana_fr_test.jsonl \
  --output-jsonl outputs/chrononoise_claims_fr.jsonl \
  --target-total-examples 20 \
  --evidence-mode both \
  --model-generation gpt-5-mini \
  --model-verification gpt-5-mini \
  --resume \
  --verbose

Export/push:

export HF_TOKEN="hf_..."

python build_claims_dataset.py \
  --output-jsonl outputs/chrononoise_claims_fr.jsonl \
  --export-only \
  --export-hf \
  --push-to-hub \
  --hub-dataset-id EmanuelaBoros/chrononoise-claims-fr \
  --private
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None


SUPPORT_LABELS = [
    "SUPPORTED",
    "INFERRED_SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "UNCERTAIN_DUE_TO_OCR",
    "TEMPORALLY_INVALID",
    "ENTITY_DRIFT",
    "OVER_SPECIFIED",
]

RISK_TYPES = [
    "NONE",
    "OCR_AMBIGUITY",
    "ENTITY_OVERLINKING",
    "LOCATION_NORMALIZATION",
    "RELATIVE_DATE_RESOLUTION",
    "NUMBER_DISTORTION",
    "TITLE_OR_ROLE_INFERENCE",
    "EVENT_OVERINTERPRETATION",
    "CAUSE_EFFECT_HALLUCINATION",
    "MODERNIZATION",
    "SUMMARY_COMPRESSION_ERROR",
    "HISTORICAL_PRIOR_HALLUCINATION",
    "UNSUPPORTED_SPECIFICATION",
]

CLAIM_GENERATION_SYSTEM_PROMPT = """You are analyzing historical newspaper text.

You will receive noisy OCR text, corrected text, or both, plus metadata such as publication date.

Your task is to produce:
1. A short factual summary.
2. A list of atomic historical claims made by the summary.

Important rules:
- Claims must be atomic: one factual statement per claim.
- Do not add background knowledge unless it is explicitly supported by the evidence.
- Do not over-resolve entities unless the evidence supports the resolution.
- Mark claims that rely on publication date or relative temporal expressions.
- If the evidence is uncertain, keep the claim cautious.
"""

CLAIM_VERIFICATION_SYSTEM_PROMPT = """You verify historical claims against noisy OCR/corrected newspaper evidence.

For each claim, decide whether it is supported by the provided evidence.

Use these labels:
- SUPPORTED: directly supported by the evidence.
- INFERRED_SUPPORTED: supported by simple reasoning using metadata, e.g. publication date + "yesterday".
- PARTIALLY_SUPPORTED: some parts are supported, others are not.
- UNSUPPORTED: not present in the evidence.
- CONTRADICTED: evidence says something different.
- UNCERTAIN_DUE_TO_OCR: OCR ambiguity prevents confident judgment.
- TEMPORALLY_INVALID: date, period, or chronology is wrong.
- ENTITY_DRIFT: entity was changed, over-resolved, or linked too specifically.
- OVER_SPECIFIED: the claim adds precision not justified by the evidence.

Focus on evidence grounding, not whether the claim is historically plausible.
A claim can be historically true but unsupported by the given text.
"""

CLAIM_GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "llm_output": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "EVENT",
                            "ENTITY",
                            "DATE",
                            "LOCATION",
                            "NUMBER",
                            "ROLE_TITLE",
                            "RELATION",
                            "OTHER",
                        ],
                    },
                    "requires_temporal_context": {"type": "boolean"},
                    "requires_entity_resolution": {"type": "boolean"},
                },
                "required": [
                    "claim_id",
                    "claim_text",
                    "claim_type",
                    "requires_temporal_context",
                    "requires_entity_resolution",
                ],
            },
        },
    },
    "required": ["llm_output", "claims"],
}

CLAIM_VERIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "support_label": {"type": "string", "enum": SUPPORT_LABELS},
                    "evidence_span": {"type": "string"},
                    "explanation": {"type": "string"},
                    "risk_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": RISK_TYPES},
                    },
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": [
                    "claim_id",
                    "claim_text",
                    "support_label",
                    "evidence_span",
                    "explanation",
                    "risk_types",
                    "confidence",
                ],
            },
        },
        "global_label": {
            "type": "string",
            "enum": [
                "fully_supported",
                "mostly_supported",
                "mixed_support",
                "mostly_unsupported",
                "uncertain",
            ],
        },
        "global_risk_summary": {"type": "string"},
    },
    "required": ["claim_verifications", "global_label", "global_risk_summary"],
}


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def ensure_output_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


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


def safe_json_loads(x: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return default
    return default


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path}", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    ensure_output_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                n += 1
            except Exception:
                continue
    return n


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("id"):
                    ids.add(obj["id"])
            except Exception:
                continue
    return ids


def make_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai is not installed. Install with: pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Use: export OPENAI_API_KEY='your_key'")
    return OpenAI()


def call_with_retries(fn, retries: int = 3, sleep_seconds: float = 5.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as err:
            last_err = err
            tqdm.write(f"[retry] Attempt {attempt + 1}/{retries} failed: {repr(err)}")
            if attempt < retries - 1:
                wait = sleep_seconds * (attempt + 1)
                tqdm.write(f"[retry] Sleeping {wait:.1f}s")
                time.sleep(wait)
    raise last_err


def generate_claims_with_openai(client, evidence_context: str, model: str, retries: int = 3) -> Dict[str, Any]:
    def _call():
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CLAIM_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": evidence_context},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chrononoise_claim_generation",
                    "schema": CLAIM_GENERATION_SCHEMA,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)

    return call_with_retries(_call, retries=retries)


def verify_claims_with_openai(
    client,
    evidence_context: str,
    generated: Dict[str, Any],
    model: str,
    retries: int = 3,
) -> Dict[str, Any]:
    user_prompt = f"""Evidence context:
{evidence_context}

Generated summary:
{generated.get("llm_output", "")}

Claims to verify:
{json.dumps(generated.get("claims", []), ensure_ascii=False, indent=2)}
"""

    def _call():
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CLAIM_VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chrononoise_claim_verification",
                    "schema": CLAIM_VERIFICATION_SCHEMA,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)

    return call_with_retries(_call, retries=retries)


def get_input_texts(record: Dict[str, Any]) -> tuple[str, str]:
    return as_string(record.get("ocr_text")), as_string(record.get("corrected_text"))


def get_source_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    if "source_metadata" in record:
        return safe_json_loads(record.get("source_metadata"), default={})
    if "source_metadata_json" in record:
        return safe_json_loads(record.get("source_metadata_json"), default={})
    return {}


def make_evidence_context(record: Dict[str, Any], evidence_mode: str, max_chars: int) -> str:
    ocr_text, corrected_text = get_input_texts(record)
    source_metadata = get_source_metadata(record)

    publication_date = as_string(record.get("date") or source_metadata.get("date"))
    title = as_string(record.get("title") or source_metadata.get("title"))
    language = as_string(record.get("language") or "fr")

    if evidence_mode == "ocr":
        evidence_text = f"OCR text:\n{ocr_text}"
    elif evidence_mode == "corrected":
        evidence_text = f"Corrected text:\n{corrected_text}"
    elif evidence_mode == "both":
        evidence_text = f"OCR text:\n{ocr_text}\n\nCorrected text:\n{corrected_text}"
    else:
        raise ValueError(f"Unknown evidence_mode: {evidence_mode}")

    if len(evidence_text) > max_chars:
        evidence_text = evidence_text[:max_chars] + "\n[TRUNCATED]"

    return f"""You are given a historical newspaper paragraph.

Metadata:
- language: {language}
- newspaper title: {title or "unknown"}
- publication date: {publication_date or "unknown"}
- evidence mode: {evidence_mode}

{evidence_text}
"""


def make_output_id(record: Dict[str, Any], evidence_mode: str) -> str:
    base_id = as_string(record.get("id") or record.get("paragraph_id") or record.get("source_id"))
    if not base_id:
        ocr_text, corrected_text = get_input_texts(record)
        base_id = stable_hash(ocr_text + corrected_text)
    return f"chrononoise_claims_{evidence_mode}_{stable_hash(base_id + evidence_mode)}"


def build_claim_record(
    input_record: Dict[str, Any],
    generated: Dict[str, Any],
    verified: Dict[str, Any],
    evidence_context: str,
    evidence_mode: str,
) -> Dict[str, Any]:
    ocr_text, corrected_text = get_input_texts(input_record)
    source_metadata = get_source_metadata(input_record)

    source_record_id = as_string(input_record.get("id"))
    title = as_string(input_record.get("title") or source_metadata.get("title"))
    publication_date = as_string(input_record.get("date") or source_metadata.get("date"))
    language = as_string(input_record.get("language") or "fr")
    output_id = make_output_id(input_record, evidence_mode)

    return {
        "id": output_id,
        "source_dataset": "ChronoCorrect-Europeana",
        "source_record_id": source_record_id,
        "language": language,
        "title": title,
        "publication_date": publication_date,
        "evidence_mode": evidence_mode,
        "ocr_text": ocr_text,
        "corrected_text": corrected_text,
        "evidence_context": evidence_context,
        "llm_output": generated.get("llm_output", ""),
        "claims": generated.get("claims", []),
        "claim_verifications": verified.get("claim_verifications", []),
        "global_label": verified.get("global_label", ""),
        "global_risk_summary": verified.get("global_risk_summary", ""),
        "source_metadata": source_metadata,
        "input_record": {
            "id": input_record.get("id"),
            "paragraph_id": input_record.get("paragraph_id"),
            "source_id": input_record.get("source_id"),
            "annotation_status": input_record.get("annotation_status"),
        },
    }


def flatten_record_for_hf(record: Dict[str, Any]) -> Dict[str, Any]:
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


def flat_hf_features() -> Features:
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


def make_hf_dataset_from_jsonl(jsonl_path: str, validation_ratio: float, test_ratio: float, seed: int) -> DatasetDict:
    records = load_jsonl(jsonl_path)
    records = [flatten_record_for_hf(r) for r in records]
    random.Random(seed).shuffle(records)

    n_total = len(records)
    if n_total < 10:
        n_test = 0
        n_val = 0
    else:
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * validation_ratio)

    test_records = records[:n_test]
    val_records = records[n_test : n_test + n_val]
    train_records = records[n_test + n_val :]

    features = flat_hf_features()
    ds = DatasetDict()
    ds["train"] = Dataset.from_list(train_records, features=features)
    if val_records:
        ds["validation"] = Dataset.from_list(val_records, features=features)
    if test_records:
        ds["test"] = Dataset.from_list(test_records, features=features)
    return ds


def write_dataset_card(out_dir: str, hub_dataset_id: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    card = f"""---
language:
- fr
task_categories:
- text-classification
- text2text-generation
pretty_name: ChronoNoise Claims
tags:
- historical-newspapers
- ocr
- hallucination-detection
- claim-verification
- historical-nlp
- cultural-heritage
- llm-generated
---

# ChronoNoise-Claims

ChronoNoise-Claims is a silver dataset for evaluating whether LLM-generated historical claims are supported by noisy OCR text, post-corrected text, and document metadata.

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

## Labels

Claim-level support labels include `SUPPORTED`, `INFERRED_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `UNCERTAIN_DUE_TO_OCR`, `TEMPORALLY_INVALID`, `ENTITY_DRIFT`, and `OVER_SPECIFIED`.

## Note

This is a silver dataset. Claims and verification labels are automatically generated and should be manually verified before use as gold data.

## Loading JSON columns

```python
import json
from datasets import load_dataset

ds = load_dataset("{hub_dataset_id}")
ex = ds["train"][0]
claims = json.loads(ex["claims_json"])
verifications = json.loads(ex["claim_verifications_json"])
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def export_and_push(args) -> None:
    if not args.export_hf and not args.push_to_hub:
        return

    tqdm.write("[hf] Building flat HF DatasetDict")
    ds = make_hf_dataset_from_jsonl(
        jsonl_path=args.output_jsonl,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    tqdm.write(str(ds))
    tqdm.write(str(ds["train"].features))

    if args.hf_output_dir:
        tqdm.write(f"[hf] Saving locally to {args.hf_output_dir}")
        ds.save_to_disk(args.hf_output_dir)
        write_dataset_card(args.hf_output_dir, args.hub_dataset_id or "EmanuelaBoros/chrononoise-claims-fr")

    if args.push_to_hub:
        if not args.hub_dataset_id:
            raise ValueError("--hub-dataset-id is required with --push-to-hub")
        if args.hf_token:
            if hf_login is None:
                raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")
            hf_login(token=args.hf_token)
        tqdm.write(f"[hf] Pushing to {args.hub_dataset_id}")
        ds.push_to_hub(
            args.hub_dataset_id,
            private=args.private,
            commit_message=args.commit_message,
        )
        tqdm.write(f"[hf] Done: https://huggingface.co/datasets/{args.hub_dataset_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChronoNoise-Claims from a ChronoCorrect JSONL file.")

    parser.add_argument("--input-jsonl", default=None, help="Input ChronoCorrect JSONL file.")
    parser.add_argument("--output-jsonl", required=True, help="Output ChronoNoise-Claims JSONL file.")
    parser.add_argument("--evidence-mode", choices=["ocr", "corrected", "both"], default="both")
    parser.add_argument("--max-evidence-chars", type=int, default=5000)
    parser.add_argument("--model-generation", default="gpt-5-mini")
    parser.add_argument("--model-verification", default="gpt-5-mini")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--target-total-examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-hf", action="store_true")
    parser.add_argument("--hf-output-dir", default="hf_dataset")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-dataset-id", default=None)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Upload ChronoNoise-Claims dataset")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_output_dir(args.output_jsonl)

    tqdm.write("[setup] Starting ChronoNoise-Claims builder")
    tqdm.write(f"[setup] Output: {args.output_jsonl}")
    tqdm.write(f"[setup] Evidence mode: {args.evidence_mode}")
    tqdm.write(f"[setup] Export only: {args.export_only}")

    if args.export_only:
        export_and_push(args)
        return

    if not args.input_jsonl:
        raise ValueError("--input-jsonl is required unless --export-only is used")

    input_records = load_jsonl(args.input_jsonl)
    existing_ids = load_existing_ids(args.output_jsonl) if args.resume else set()
    existing_count = count_jsonl_records(args.output_jsonl) if args.resume else 0

    if args.target_total_examples is not None:
        remaining = max(args.target_total_examples - existing_count, 0)
    else:
        remaining = args.max_examples

    tqdm.write(f"[setup] Input records: {len(input_records)}")
    tqdm.write(f"[setup] Existing output records: {existing_count}")
    tqdm.write(f"[setup] Remaining to write: {remaining}")

    if remaining == 0:
        tqdm.write("[setup] Target already reached.")
        export_and_push(args)
        return

    client = None
    if not args.no_api:
        client = make_openai_client()

    written = 0
    skipped_existing = 0
    errors = 0

    pbar = tqdm(total=remaining, desc="Building ChronoNoise-Claims", unit="example")

    for input_record in input_records:
        if written >= remaining:
            break

        output_id = make_output_id(input_record, args.evidence_mode)
        if output_id in existing_ids:
            skipped_existing += 1
            continue

        evidence_context = make_evidence_context(
            input_record,
            evidence_mode=args.evidence_mode,
            max_chars=args.max_evidence_chars,
        )

        if args.verbose:
            tqdm.write(f"[record] {output_id}")

        if args.no_api:
            generated = {"llm_output": "", "claims": []}
            verified = {
                "claim_verifications": [],
                "global_label": "uncertain",
                "global_risk_summary": "No API mode.",
            }
        else:
            try:
                pbar.set_postfix({"stage": "generate_claims", "written": written})
                generated = generate_claims_with_openai(
                    client=client,
                    evidence_context=evidence_context,
                    model=args.model_generation,
                    retries=args.api_retries,
                )
                pbar.set_postfix({"stage": "verify_claims", "written": written})
                verified = verify_claims_with_openai(
                    client=client,
                    evidence_context=evidence_context,
                    generated=generated,
                    model=args.model_verification,
                    retries=args.api_retries,
                )
            except Exception as err:
                errors += 1
                tqdm.write(f"[error] {output_id}: {repr(err)}")
                if args.resume:
                    continue
                raise

        record = build_claim_record(
            input_record=input_record,
            generated=generated,
            verified=verified,
            evidence_context=evidence_context,
            evidence_mode=args.evidence_mode,
        )

        write_jsonl_record(args.output_jsonl, record)
        existing_ids.add(output_id)
        written += 1
        pbar.update(1)
        pbar.set_postfix({"stage": "written", "written": written, "errors": errors})

        if args.sleep > 0:
            time.sleep(args.sleep)

    pbar.close()

    print("\nDone.")
    print(f"Existing records before run: {existing_count}")
    print(f"New records written: {written}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Errors: {errors}")
    print(f"Output: {args.output_jsonl}")

    export_and_push(args)


if __name__ == "__main__":
    main()
