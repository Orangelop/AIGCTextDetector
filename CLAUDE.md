# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Text Detector — client-side Chinese text classifier that detects AI-generated Chinese creative writing (web novels/fan fiction). Uses 7 binary TF-IDF + LinearSVC models (one per source AI model) running entirely in the browser.

Online demo: https://lyc8503.github.io/AITextDetector/

## Code Architecture

Three directories, each a distinct pipeline stage:

### `srctrain/` — Model Training (Python)

- **loader.py** — Loads `chapters_sample.json` (human-written Chinese novels), pairs with AI-generated chapters from `generated_*/` dirs. Splits by novel ID into train/test sets. Contains `split_chinese_sentence()` for sentence segmentation and `to_col()` for converting samples into feature-label pairs.
- **train.py** — Multi-class TF-IDF + LinearSVC baseline training & evaluation.
- **train_binary.py** — Two-stage training pipeline: (1) full TF-IDF (char 2-5 grams, min_df=3, sublinear_tf) + LinearSVC, (2) prunes to top-K features (default 20k) by |idf · coef|, retrains restricted model. Exports joblib files to `model/`. Trains all 7 binary models in parallel via ProcessPoolExecutor.
- **export_for_web.py** — Loads pruned joblib models, exports feature weights (combined = idf · coef) and intercepts to `srcweb/models.json` (~107MB). Used by GitHub Pages deployment.

### `srcgen/` — Training Data Generation (Python)

- **summarize.py** — Uses LLM API to summarize human-written chapter content into ~500-char summaries.
- **gen.py** — Uses LLM API to expand summaries back into full chapters (2000 chars each). Supports streaming + OpenTelemetry metrics. Configurable model, batch size, parallelism.
- **metrics.py** — OpenTelemetry counters (token usage, output chars) exported to Grafana OTLP endpoint.

### `srcweb/` — Browser App (HTML + JS)

- **index.html** — Single self-contained page. Client-side inference:
  1. Loads `models.json` (~107MB) from cache or network with progress bar
  2. User inputs text (or pastes LOFTER fanfic link for proxy-based fetch)
  3. Splits text on sentence boundaries, cleans non-Chinese chars
  4. Counts char n-grams (2-5), runs each against all 7 models
  5. Replicates sklearn's `sublinear_tf=True`, `norm='l2'` LinearSVC decision function
  6. Renders highlighted text with per-sentence and per-model breakdowns
  7. Shows verdict (Human / Maybe Human / Maybe AI) based on % AI-flagged chars

### GitHub Actions (`.github/workflows/pages.yml`)

Deploys `srcweb/` to GitHub Pages on push to master. Downloads `models.json` from GitHub Releases during build.

## Key Models (7 binary classifiers)

gemini, qwen, pony, kimi25, glm47, doubao, deepseekv32

## Requirements (Python)

scikit-learn, numpy, joblib, openai, tqdm, opentelemetry-sdk, opentelemetry-exporter-otlp-proto-http (optional, for srcgen metrics)
