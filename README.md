# ChronoNoise-Claims Dataset

**ChronoNoise-Claims** is a dataset-construction pipeline for studying how large language models generate and verify historical claims from noisy OCR and post-corrected historical newspaper text.

The project is designed around one question:

> When an LLM reads noisy or corrected historical text, can it distinguish what is actually supported by the document from what is merely historically plausible?

The repository currently contains scripts to build a silver claim-verification dataset from ChronoCorrect-Europeana-style JSONL files.

---

Historical newspaper OCR often contains character errors, broken words, missing accents, layout noise, and entity distortions. When LLMs are asked to summarize or interpret such text, they may produce fluent historical claims that sound plausible but are not actually supported by the evidence.

This project focuses on the chain:

```text
historical OCR noise → LLM interpretation → evidence-grounded historical claims
```

ChronoNoise-Claims is intended to support research on:

- OCR-sensitive hallucination detection
- claim verification over noisy historical documents
- evidence-grounded historical NLP
- temporal grounding and relative-date interpretation
- entity drift caused by OCR or overcorrection
- downstream effects of OCR post-correction on LLM outputs

---

## Dataset idea

Each example starts from a historical newspaper paragraph with:

- original OCR text
- corrected text
- metadata such as language, title, and publication date

The pipeline asks an LLM to produce a short historical summary and a set of atomic claims. A second verification step then labels each claim according to whether it is supported by the provided evidence.

Example labels include:

- `SUPPORTED`
- `INFERRED_SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `CONTRADICTED`
- `UNCERTAIN_DUE_TO_OCR`
- `TEMPORALLY_INVALID`
- `ENTITY_DRIFT`
- `OVER_SPECIFIED`

Risk types include:

- `OCR_AMBIGUITY`
- `ENTITY_OVERLINKING`
- `LOCATION_NORMALIZATION`
- `RELATIVE_DATE_RESOLUTION`
- `NUMBER_DISTORTION`
- `TITLE_OR_ROLE_INFERENCE`
- `EVENT_OVERINTERPRETATION`
- `CAUSE_EFFECT_HALLUCINATION`
- `HISTORICAL_PRIOR_HALLUCINATION`
- `UNSUPPORTED_SPECIFICATION`

---

---

## Input format

The builder expects a JSONL file produced by a ChronoCorrect-Europeana-style pipeline.

Each input record should contain at least:

```json
{
  "id": "chronocorrect_europeana_fr_...",
  "language": "fr",
  "title": "...",
  "date": "...",
  "ocr_text": "...",
  "corrected_text": "...",
  "source_metadata": {...}
}
```

The script is also compatible with flattened fields such as `source_metadata_json`.

---

## Quick no-API test

Run a small dry test first:

```bash
mkdir -p outputs

python build_claims_dataset.py \
  --input-jsonl ../europeana-post-correct-data/outputs/chronocorrect_europeana_fr_test.jsonl \
  --output-jsonl outputs/test_no_api_claims.jsonl \
  --max-examples 5 \
  --no-api \
  --resume \
  --verbose
```

This creates placeholder examples without calling the API.

---

## Build the claims dataset

Example run for 20 examples:

```bash
mkdir -p outputs
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
```

The `--resume` flag skips examples already present in the output JSONL.

The `--target-total-examples` flag means: continue until the output JSONL contains this many valid examples in total.

For example, if 20 examples already exist and you run:

```bash
--target-total-examples 100
```

then the script writes 80 new examples.

---

## Evidence modes

The script supports three evidence modes:

```bash
--evidence-mode ocr
--evidence-mode corrected
--evidence-mode both
```

These modes allow comparison between LLM behavior on:

1. noisy OCR only
2. corrected text only
3. OCR and corrected text together

This is useful for analyzing whether OCR correction improves factual grounding or sometimes increases overconfident unsupported claims.

---

## Output format

Each output record contains:

```json
{
  "id": "chrononoise_claims_both_...",
  "source_dataset": "ChronoCorrect-Europeana",
  "source_record_id": "...",
  "language": "fr",
  "title": "...",
  "publication_date": "...",
  "evidence_mode": "both",
  "ocr_text": "...",
  "corrected_text": "...",
  "evidence_context": "...",
  "llm_output": "...",
  "claims": [...],
  "claim_verifications": [...],
  "global_label": "mostly_supported",
  "global_risk_summary": "...",
  "source_metadata": {...},
  "input_record": {...}
}
```

A claim verification item looks like:

```json
{
  "claim_id": "c1",
  "claim_text": "The meeting took place in Paris.",
  "support_label": "SUPPORTED",
  "evidence_span": "...",
  "explanation": "The claim is directly stated in the corrected paragraph.",
  "risk_types": ["NONE"],
  "confidence": "high"
}
```

---

## How to use it?

### Claim verification

Input:

```text
historical paragraph + generated claim
```

Output:

```text
support label + evidence span + risk type
```

### OCR-sensitive hallucination detection

Input:

```text
noisy OCR + LLM summary
```

Output:

```text
unsupported or risky claims
```

### Evidence selection

Input:

```text
paragraph + claim
```

Output:

```text
evidence span supporting or contradicting the claim
```

### Temporal grounding

Input:

```text
publication date + historical text + relative temporal expression
```

Output:

```text
supported temporal interpretation or temporal-risk label
```

---

## Limitations

The current dataset is silver. 

---

## Citation

If you use this repository or the derived dataset, please cite the original source dataset and this derived dataset/repository.

Suggested citation placeholder:

```bibtex
@dataset{boros_chrononoise_claims,
  title = {ChronoNoise-Claims: Evidence-Grounded Claim Verification over Noisy Historical Newspaper Text},
  author = {Boros, Emanuela},
  year = {2026},
  publisher = {Hugging Face / GitHub},
  note = {Silver dataset construction pipeline for historical OCR-sensitive claim verification}
}
```




