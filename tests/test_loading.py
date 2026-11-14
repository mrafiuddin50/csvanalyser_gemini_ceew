"""Reading CSV, TSV, TXT and Excel files."""
import io

import pandas as pd

from csvanalyser import loading

# File loading

def test_utf8_comma():
    df, desc, _ = loading.load_table("a.csv", "site,pm25\nAnand Vihar,1.5\n".encode("utf-8"))
    assert list(df.columns) == ["site", "pm25"] and desc == "UTF-8, comma-separated"


def test_utf8_bom():
    df, desc, _ = loading.load_table("a.csv", b"\xef\xbb\xbfsite,pm25\nA,1\n")
    assert list(df.columns) == ["site", "pm25"] and desc.startswith("UTF-8 with BOM")


def test_windows_1252_semicolon():
    raw = "site;temp °C;note\nKöln;12.5;“quoted”\nDelhi;30.1;ok\n".encode("cp1252")
    df, desc, _ = loading.load_table("a.csv", raw)
    assert list(df.columns) == ["site", "temp °C", "note"]
    assert df["site"][0] == "Köln" and desc == "Windows-1252, semicolon-separated"


def test_latin1_fallback():
    raw = b"site,val\n\x81\x8d,1\n"   # bytes undefined in cp1252
    df, desc, _ = loading.load_table("a.csv", raw)
    assert desc.startswith("Latin-1") and len(df) == 1


def test_tab_and_pipe():
    df, desc, _ = loading.load_table("a.tsv", b"a\tb\n1\t2\n")
    assert list(df.columns) == ["a", "b"] and "tab" in desc
    df, desc, _ = loading.load_table("a.txt", b"a|b|c\n1|2|3\n4|5|6\n")
    assert list(df.columns) == ["a", "b", "c"] and "pipe" in desc


def test_commas_inside_quotes_and_semicolons_in_text():
    raw = b'site,note\n"Okhla, Phase 2","a;b;c"\nDelhi,"x;y"\n'
    df, desc, _ = loading.load_table("a.csv", raw)
    assert list(df.columns) == ["site", "note"] and df["site"][0] == "Okhla, Phase 2"


def test_single_column():
    df, desc, _ = loading.load_table("a.csv", b"value\n1\n2\n")
    assert list(df.columns) == ["value"]


def test_excel(tmp_path):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame({"y": [3]}).to_excel(writer, sheet_name="Notes", index=False)
    df, desc, _ = loading.load_table("book.xlsx", buf.getvalue())
    assert list(df.columns) == ["x"] and "first of 2 sheets" in desc
