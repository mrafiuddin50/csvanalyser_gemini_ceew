# Universal CSV Analyser

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
- A local data profile
- Editable code, with an "Explain this code" button
- Downloads: results as CSV, and the session as a Python script or Jupyter notebook
- A synthetic sample dataset for practice

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
| `index.html` | Loads stlite and lists every Python file the app uses |
| `stlite_app.py` | The Streamlit page |
| `csvanalyser/` | The app's logic: Gemini client, privacy rules and prompts, the sealed-off code runner, file loading, data profile, sample data, exports |
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

The benchmark asks 16 questions with known answers about two synthetic datasets and scores each model's answers. It needs your own Gemini API key, read from an environment variable. Never put the key in a file in this repository.

```bash
export GEMINI_API_KEY=your-key-here
python -m benchmark.run_benchmark --models gemini-2.5-flash,gemini-2.5-pro
```

The run prints accuracy, retries and time per model, and saves detailed results in `benchmark/results/`, which git ignores. Each question uses one to three API requests per model.

To check the benchmark itself without a key or network access, run:

```bash
python -m benchmark.run_benchmark --self-check
```
