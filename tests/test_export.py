"""Exported scripts and notebooks."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from csvanalyser import analysis, export, loading


def make_df():
    return pd.DataFrame({
        "site_name": ["Anand Vihar", "Punjabi Bagh", "Okhla Phase-2", "Anand Vihar"],
        "pm25": [123.45, 678.91, None, 123.45],
        "date": ["2023-01-05", "2023-02-17", "2023-03-09", "2023-01-05"],
    })


def fake_llm(code):
    return lambda prompt: code


# Exports

def _session(df):
    t1 = analysis.analyse("mean pm25 by site", df, "", [], False,
                     fake_llm("st.subheader('Means')\nst.bar_chart(df.groupby('site_name')['pm25'].mean())\n"
                              "fig, ax = plt.subplots(); ax.plot([1, 2]); st.pyplot(fig)\n"
                              "result = df.groupby('site_name')['pm25'].mean()"))
    t2 = analysis.analyse("rows?", df, "", [t1], True, fake_llm("result = len(df[df['col_2'].notna()])"))
    t3 = analysis.run_edited_code(t1, "result = df['pm25'].max()", df)
    bad = analysis.analyse("fails", df, "", [], False, fake_llm("result = df['nope']"))
    return [t1, t2, bad, t3]


@pytest.fixture
def data_dir(tmp_path):
    raw = "site_name;pm25;date\nKöln;12.5;2023-01-05\nDelhi;30.1;2023-01-06\nDelhi;;2023-01-07\n".encode("cp1252")
    (tmp_path / "my data.csv").write_bytes(raw)
    df, _, reader = loading.load_table("my data.csv", raw)
    assert reader == "pd.read_csv('my data.csv', sep=';', encoding='cp1252')"
    return tmp_path, df, reader


def test_script_export_runs(data_dir):
    tmp, df, reader = data_dir
    script = export.build_script(_session(df), reader, df.columns)
    assert "fails" not in script                      # failed turn left out
    assert "(edited by you)" in script and "ALIASES = {" in script
    assert "Köln" not in script and "12.5" not in script   # no data or results in the file
    (tmp / "analysis_session.py").write_text(script, encoding="utf-8")
    proc = subprocess.run([sys.executable, "analysis_session.py"], cwd=tmp, capture_output=True, text=True,
                          env={**os.environ, "MPLBACKEND": "Agg"}, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "Means" in proc.stdout and "30.1" in proc.stdout and "2" in proc.stdout


def test_notebook_export_is_valid_and_runs(data_dir):
    tmp, df, reader = data_dir
    nb = json.loads(export.build_notebook(_session(df), reader, df.columns))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 4 and all(c["outputs"] == [] for c in code_cells)
    try:
        import nbformat
        nbformat.validate(nbformat.reads(json.dumps(nb), as_version=4))
    except ImportError:
        pass
    # Execute the cells in order, as Jupyter would.
    import matplotlib
    matplotlib.use("Agg")
    shown = []
    namespace = {"display": shown.append}
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        for c in code_cells:
            exec("".join(c["source"]), namespace)
    finally:
        os.chdir(cwd)
    assert namespace["result"] == 30.1
    assert "Means" in shown
