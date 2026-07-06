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


