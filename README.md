# Tabular Data Analyser

https://mrafiuddin50.github.io/csvanalyser_gemini_ceew/

A simple tool to analyse data from CSV files using Google's Gemini models. The primary purpose is to train researchers at CEEW.

This tool has been inspired by the VayuChat-WebLLM tool by IIT Gandhinagar's Sustainability Lab.
https://github.com/nipunbatra/vayuchat-webllm

## How it works

The app runs in the browser using [stlite](https://github.com/whitphx/stlite) (Streamlit on WebAssembly). You upload a CSV or Excel file and ask a question in plain English. Gemini writes Python code for the question, and that code runs on your own computer, in the browser.

**The model never sees your data.** Only your question, the context instructions, and the column names and types are sent to Gemini. Rows, values, results and charts stay in the browser. Error messages are cleaned of data values before a retry, and the generated code cannot reach the internet. The "How your data is handled" section in the app has the details, including how Google treats requests from free-tier and paid API keys.

Features:

- Self-correcting retries when generated code fails
- Follow-up questions
- Optional anonymised column names
- A **🧭 Take the tour** button: a guided tour that moves to each part of the page, highlights it and explains
  it (switching tabs as needed). It is built with [driver.js](https://driverjs.com) (MIT licence) in `tour.js`,
  and runs only on request
- A printable user guide at [`guide.html`](https://mrafiuddin50.github.io/csvanalyser_gemini_ceew/guide.html),
  with practice exercises for training sessions
- Three main tabs: **① Data & quality**, **② Ask questions** (with example questions built from the column
  names) and **③ Export**
- Answers as cards with **Result / Code / Sent to Gemini** tabs
- CEEW brand colours on a white page (blue family only, following the brand rule of not mixing primary colours)
- An automatic data quality audit when a file loads (see below)
- A local data profile
- Editable code, with an "Explain this code" button
- Charts in the CEEW house style, following the CEEW data viz skill (see below)
- Downloads: results as CSV, charts as PNG (300 dpi) or SVG, and the session as a Python script or Jupyter notebook
- A synthetic sample dataset for practice

## CEEW charts

Charts follow the CEEW data viz skill: its chart menu, rules R1 to R8, colour rules and house style
(`csvanalyser/charts.py`). The skill assumes the AI sees the data; this app's rule is that Gemini never does, so
the work is split:

- **Gemini** chooses a chart function (`ceew.bar`, `ceew.line`, `ceew.pie`, …) from the question and the column
  names, and gives a one-line reason. The function list and the rules it must follow are in `csvanalyser/prompts.py`.
- **The browser** checks the rules that need the data and falls back with a note when one fails: parts must add
  up to 100% for pie and stacked 100% charts, bars start at zero and hold at most 15 categories, lines break at
  gaps, more than eight categories turn grey, highlights only when the question names them. These messages are
  shown to the researcher and never sent to Gemini.
- **The researcher** writes the headline, source and note under each chart; each appears on the chart only once it
  has been entered, and no space is kept for it until then. The headline is checked locally for words that claim a cause or a record; it is never rewritten
  and never sent to Gemini (a headline states findings from the data).

The sidebar's *4 · Chart style* sets the use (research report, presentation, outreach) and the colour pair.
Charts are drawn with matplotlib in Familjen Grotesk (`assets/fonts/`, SIL Open Font License, from
[Familjen-Sthlm/Familjen-Grotesk](https://github.com/Familjen-Sthlm/Familjen-Grotesk)). The Export tab offers
`ceew_charts.py` so exported notebooks can draw the same charts.

Not included yet: waterfall, treemap, waffle, Sankey and the other GUARDRAIL charts beyond the common ones,
small multiples, and the outreach colour-blind check.

## Data quality audit

When a file loads, the app audits it in the browser (`csvanalyser/audit.py`), following the method of the
*data-quality-auditor-v2* skill. The audit is rule-based and calculated on the full dataset. No AI is involved,
and nothing from the audit is sent to Gemini.

- **Banner:** gives the overall assessment (Good / Needs Attention / Poor) and the analysis readiness (READY /
  READY WITH CAUTION / NOT READY).
- **Data quality audit tab:** found under *Your data*, it contains:
  - findings, split into critical issues, warnings and observations, each with affected records and impact;
  - recommended actions;
  - a column-level summary and statistics;
  - **What you can analyse**: whether counts, averages, group comparisons, trends over time, correlations,
    joining, maps and sharing are ready, need caution or are not possible, and why.
- **HTML report:** a downloadable copy of the audit. To get a PDF, open it in a browser and use Print → Save as
  PDF.

The checks cover:

- missing values and placeholder text;
- duplicate rows and repeated IDs;
- numbers or dates stored as text, mixed types, unreadable dates and day/month order;
- impossible coordinates, out-of-range percentages and negative quantities (flagged for confirmation);
- inconsistent category labels and rare categories;
- possible personal data (emails, phone numbers, Aadhaar- and PAN-like patterns). These are counted only,
  never shown;
- outliers (1.5 × IQR), skew and strong correlations;
- for time series, conflicting records per entity and date, and gaps in reporting.

The audit is general-purpose and does not know the subject area. Treat its flags as prompts to check with someone
who knows the data, not as proof of errors.

## Memory use

Running Python in the browser is memory-hungry. In one measurement (November 2026, stlite 1.12.0), the browser
tab used:

- about 510 MB for an empty Streamlit page under stlite;
- about 645 MB for this app, just after loading.

Other browsers report different figures; 1.1 GB has been seen. So most of the cost is stlite itself, not the
app. Memory also grows as larger files are analysed, and it is only released when the tab is reloaded or
closed.

The app shows users a "Memory and performance tips" section:

- use one tab;
- reload after heavy sessions, exporting results first;
- filter very large files before uploading;
- close other heavy tabs on low-memory computers.

For a large reduction, the page would need rebuilding without Streamlit, as plain HTML/JavaScript with Pyodide
running pandas. A test page with Pyodide and pandas alone used about 280 MB.

## Project layout

| Path | What it is |
| --- | --- |
| `index.html` | Loads stlite, sets the theme and lists every Python file the app uses |
| `guide.html`, `guide-images/` | The printable user guide and its screenshots |
| `tour.js`, `tour.css` | The guided tour. Steps find page sections by their `tour-…` container keys in `stlite_app.py`; `tests/test_tour_anchors.py` checks the two match |
| `.streamlit/config.toml` | The same theme, for running locally with plain Streamlit |
| `assets/fonts/` | Familjen Grotesk, the CEEW chart font (SIL Open Font License, `OFL.txt`) |
| `stlite_app.py` | The Streamlit page |
| `csvanalyser/` | The app's logic: Gemini client, privacy rules and prompts, the sealed-off code runner, file loading, data quality audit, data profile, sample data, exports |
| `tests/` | Automated tests (no API key needed; Gemini is faked) |
| `benchmark/` | Known-answer questions for comparing models and prompts |

When you add a Python file to `csvanalyser/`, also add it to the `files` list in `index.html`. Otherwise the live app cannot load it.

## Running locally

From the repository folder, serve it with any static web server, then open http://localhost:8765/index.html:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

## Tests

Use Python 3.12:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

`tests/test_privacy_leaks.py` checks the core rule. It plants marker values in a dataset and fails if any of them appears in anything sent to Gemini. GitHub Actions runs all tests on every push (`.github/workflows/tests.yml`).

The tests run in normal Python, not in the browser. After changing `index.html` or the stlite version, also check the app in a browser.

## Benchmark

The benchmark scores each model on two synthetic datasets, in two suites:

- **answers**: 16 questions with one known number or label;
- **charts**: 16 chart requests, including hourly monitor data stored as text (average by hour of day, a vertical box plot by station), scored on the CEEW chart rules. An answer counts as correct only if it draws a fitting kind of chart with the right numbers, writes no title, chooses no colours, uses no second axis, and highlights only what the question names. Two requests ask for things the rules refuse (an AI-written title, a chart with two y-axes).

It needs your own Gemini API key, read from an environment variable. Never put the key in a file in this repository.

```bash
export GEMINI_API_KEY=your-key-here
python -m benchmark.run_benchmark --models gemini-2.5-flash,gemini-2.5-pro
python -m benchmark.run_benchmark --suite charts      # only the chart requests
```

The run prints accuracy, retries and time per model and suite, and saves detailed results (with the problems found in each chart answer) in `benchmark/results/`, which git ignores. Each question uses one to three API requests per model.

To check the benchmark itself without a key or network access, run:

```bash
python -m benchmark.run_benchmark --self-check
```
