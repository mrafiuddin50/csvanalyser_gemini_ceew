import os
import pathlib
import sys

# Make the csvanalyser package and stlite_app.py importable from the tests.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Matplotlib must not try to open windows during tests.
os.environ.setdefault("MPLBACKEND", "Agg")
