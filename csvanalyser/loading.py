"""Reading uploaded CSV, TSV, TXT and Excel files."""

import codecs
import csv
import io

import pandas as pd


UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx"]


DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}


def decode_bytes(raw: bytes):
    """Decodes text, trying UTF-8 first, then the common Windows/Latin encodings."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8):].decode("utf-8"), "UTF-8 with BOM"
    for encoding, label in (("utf-8", "UTF-8"), ("cp1252", "Windows-1252")):
        try:
            return raw.decode(encoding), label
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "Latin-1"


def sniff_delimiter(text: str, default: str = ",") -> str:
    """Picks the delimiter that splits the first lines into a consistent number of columns."""
    lines = [line for line in text.splitlines()[:50] if line.strip()][:20]
    if not lines:
        return default
    best, best_score = default, (0.0, 0)
    for delimiter in [default] + [d for d in DELIMITER_NAMES if d != default]:
        try:
            widths = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        except csv.Error:
            continue
        if not widths or widths[0] < 2:
            continue
        score = (sum(w == widths[0] for w in widths) / len(widths), widths[0])
        if score > best_score:
            best, best_score = delimiter, score
    return best


ENCODING_CODECS = {"UTF-8": None, "UTF-8 with BOM": "utf-8-sig", "Windows-1252": "cp1252", "Latin-1": "latin-1"}


def load_table(name: str, raw: bytes):
    """Reads an uploaded CSV/TSV/TXT or Excel file.

    Returns (df, description, reader_code), where reader_code is the pandas call
    that reads the same file, used in exported scripts and notebooks.
    """
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm")):
        with pd.ExcelFile(io.BytesIO(raw)) as book:
            sheets = book.sheet_names
            df = book.parse(sheets[0])
        description = f"Excel sheet '{sheets[0]}'"
        if len(sheets) > 1:
            description += f" (only the first of {len(sheets)} sheets is used)"
        return df, description, f"pd.read_excel({name!r}, sheet_name={sheets[0]!r})"

    text, encoding = decode_bytes(raw)
    delimiter = sniff_delimiter(text, default="\t" if lower.endswith(".tsv") else ",")
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    args = [repr(name)]
    if delimiter != ",":
        args.append(f"sep={delimiter!r}")
    if ENCODING_CODECS[encoding]:
        args.append(f"encoding={ENCODING_CODECS[encoding]!r}")
    return df, f"{encoding}, {DELIMITER_NAMES[delimiter]}-separated", f"pd.read_csv({', '.join(args)})"
