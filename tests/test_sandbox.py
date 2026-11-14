"""Generated code: allowed imports, removed builtins and the network block."""
import urllib.error
import urllib.request

import pandas as pd
import pytest

from csvanalyser import analysis, sandbox


def run(code, df=None):
    df = df if df is not None else pd.DataFrame({"a": [1, 2]})
    return sandbox.run_generated_code(code, df, {})


# Imports, builtins and network

@pytest.mark.parametrize("code", [
    "import urllib.request",
    "import sys",
    "import os",
    "import socket",
    "import js",
    "import pyodide",
    "import importlib",
    "from http import client",
    "import requests",
])
def test_blocked_imports(code):
    with pytest.raises(ImportError, match="not allowed"):
        run(code + "\nresult = 1")


def test_allowed_imports_work():
    result, _ = run("import difflib, math, statistics, datetime, re, collections\n"
                    "from collections import Counter\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
                    "import pandas\nresult = math.sqrt(4)")
    assert result == 2


def test_open_is_removed():
    with pytest.raises(NameError):
        run("open('/etc/passwd')\nresult = 1")


def test_indirect_network_is_blocked():
    # pandas would normally fetch this URL itself
    with pytest.raises(sandbox.NetworkBlockedError):
        run("result = pd.read_csv('http://127.0.0.1:9/x.csv')")
    with pytest.raises(sandbox.NetworkBlockedError):
        run("result = pd.io.common.urlopen('http://127.0.0.1:9/')")


def test_network_restored_after_run():
    original = urllib.request.urlopen
    with pytest.raises(sandbox.NetworkBlockedError):
        run("result = pd.read_csv('http://127.0.0.1:9/x.csv')")
    assert urllib.request.urlopen is original


def test_blocked_import_triggers_retry_with_clear_message():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "import os\nresult = 1" if len(prompts) == 1 else "result = 1"
    turn = analysis.analyse("q", pd.DataFrame({"a": [1]}), "", [], False, llm)
    assert turn["attempts"] == 2 and turn["error"] is None
    assert "ImportError: Importing os is not allowed here" in prompts[1]
