"""Data quality audit, following the method of the data-quality-auditor-v2 skill.

Runs entirely in the browser with fixed rules; nothing here is sent to Gemini.
Every number in the report is calculated from the full dataset (no sampling).
Unusual values are reported as things to check, not as errors, unless a rule
is clear-cut, and the report never repeats values from columns that look like
personal data.
"""
import datetime as dt
import html
import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CRITICAL, WARNING, OBSERVATION = "Critical", "Warning", "Observation"
READY, CAUTION, NOT_READY = "READY", "READY WITH CAUTION", "NOT READY"
CAN, CAREFUL, CANNOT = "Ready", "Use with caution", "Not possible"

# Text that usually means "no value". pandas already reads NA, N/A, null, NaN and
# empty cells as missing when the file is loaded; these are the ones it does not.
PLACEHOLDERS = {"-", "--", "---", "?", "??", ".", "missing", "nil", "not available", "unknown", "none", "n.a.",
                "na", "n/a", "nan", "null", "#n/a", "not applicable", "no data", "blank"}
BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f"}

ID_WORDS = {"id", "uid", "uuid", "key", "code", "sno", "serial"}
GEO_WORDS = {"lat", "latitude", "lon", "lng", "long", "longitude", "state", "district", "city", "country",
             "village", "block", "pincode", "pin", "zip", "postcode", "region", "ward", "tehsil", "taluk", "town"}
DATE_WORDS = {"date", "time", "timestamp", "datetime", "day", "month", "year", "dob", "period"}
PERCENT_WORDS = {"pct", "percent", "percentage"}
NON_NEGATIVE_WORDS = {"count", "number", "num", "qty", "quantity", "population", "households", "size", "age",
                      "income", "price", "cost", "amount", "kwh", "pm25", "pm10", "pm2", "concentration",
                      "rainfall", "consumption", "area", "weight", "height", "length", "volume", "capacity"}
ENTITY_WORDS = {"station", "site", "location", "id", "district", "state", "city", "household", "plant", "unit",
                "village", "sensor", "monitor", "entity", "name", "code", "region", "block", "facility", "company"}
PERSON_WORDS = {"respondent", "person", "first", "last", "full", "beneficiary", "farmer", "owner", "head",
                "applicant", "member", "father", "mother", "patient", "student", "employee", "customer"}
SENSITIVE_WORDS = {"email", "phone", "mobile", "address", "dob", "birth", "aadhaar", "aadhar", "pan", "contact",
                   "surname", "voter", "passport"}

SENSITIVE_PATTERNS = {
    "email addresses": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "phone numbers": re.compile(r"^(\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}$"),
    "Aadhaar-like numbers": re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
    "PAN-like numbers": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
}
DMY_PATTERN = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")


@dataclass
class Finding:
    severity: str
    issue: str
    impact: str
    affected: int | None = None
    total: int | None = None
    kind: str = "structure"     # "invalid", "anomaly", "domain" or "structure"
    action: str = ""
    columns: tuple = ()

    @property
    def pct(self):
        if self.affected is None or not self.total:
            return None
        return 100 * self.affected / self.total


@dataclass
class ColumnInfo:
    name: str
    storage: str
    semantic: str
    missing: int
    placeholders: int
    unique: int
    sensitive: bool = False
    day_month: str | None = None          # for d/m/y-style text dates
    findings: list = field(default_factory=list)
    numeric: pd.Series | None = None      # values as numbers (also for numbers stored as text)
    dates: pd.Series | None = None        # values as dates (also for dates stored as text)
    labels: pd.Series | None = None       # trimmed, lower-cased text labels


@dataclass
class AuditReport:
    file_name: str
    file_type: str
    rows: int
    columns: int
    column_info: list
    findings: list
    assessment: str
    assessment_text: str
    readiness: str
    readiness_text: str
    actions: list
    capabilities: list
    stats: pd.DataFrame
    correlations: pd.DataFrame
    checks_done: list
    checks_skipped: list
    generated_at: str
    identifiers: list
    temporal: list

    def by_severity(self, severity):
        return [f for f in self.findings if f.severity == severity]


# ---------------------------------------------------------------------------
# Step 1: work out what each column is
# ---------------------------------------------------------------------------

def _words(name) -> set:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}


def _is_text(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    return (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype))


def _parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    """Parses each distinct text value once, then maps back."""
    uniques = pd.Series(values.dropna().unique())
    if uniques.empty:
        return pd.Series(pd.NaT, index=values.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(uniques, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):
            parsed = pd.to_datetime(uniques, errors="coerce", dayfirst=dayfirst)
    return values.map(dict(zip(uniques, parsed)))


def _day_month_order(values: pd.Series):
    """Returns 'dayfirst', 'monthfirst', 'ambiguous', 'mixed' or None for d/m/y-style dates."""
    matches = values.dropna().str.extract(DMY_PATTERN).dropna()
    if len(matches) < max(3, 0.5 * values.notna().sum()):
        return None
    first, second = matches[0].astype(int), matches[1].astype(int)
    day_first, month_first = (first > 12).any(), (second > 12).any()
    if day_first and month_first:
        return "mixed"
    if day_first:
        return "dayfirst"
    if month_first:
        return "monthfirst"
    return "ambiguous"


def _profile_column(name, series: pd.Series, n: int) -> ColumnInfo:
    words = _words(name)
    info = ColumnInfo(name=str(name), storage=str(series.dtype), semantic="", missing=int(series.isna().sum()),
                      placeholders=0, unique=int(series.nunique(dropna=True)))
    present = series.dropna()

    if len(present) == 0:
        info.semantic = "empty"
        return info

    if pd.api.types.is_bool_dtype(series):
        info.semantic = "boolean"
        return info

    if pd.api.types.is_datetime64_any_dtype(series):
        info.semantic = "datetime"
        info.dates = series
        return info

    if pd.api.types.is_numeric_dtype(series):
        info.numeric = series.astype(float)
        values = present.astype(float)
        whole = bool(np.all(np.mod(values, 1) == 0))
        if words & {"lat", "latitude", "lon", "lng", "longitude"}:
            info.semantic = "geographic (coordinate)"
        elif set(values.unique()) <= {0.0, 1.0}:
            info.semantic = "boolean (0/1)"
        elif whole and words & {"year"} and values.between(1800, 2200).all():
            info.semantic = "datetime (year)"
        elif whole and info.unique == len(present) and len(present) >= 20 and (
                words & ID_WORDS or str(name).lower().endswith("id")
                or (values.is_monotonic_increasing and bool((np.diff(values.to_numpy()) == 1).all()))):
            info.semantic = "identifier"
        elif whole and info.unique <= 12 and len(present) >= 50 and not words & NON_NEGATIVE_WORDS:
            info.semantic = "numeric (possibly a category code)"
        else:
            info.semantic = "quantitative"
        return info

    # Text columns
    text = present.astype(str).str.strip()
    placeholder_mask = text.str.lower().isin(PLACEHOLDERS) | (text == "")
    info.placeholders = int(placeholder_mask.sum())
    clean = text[~placeholder_mask]
    labels = series.astype("object").where(series.notna()).astype(str).str.strip().str.lower()
    labels = labels.where(series.notna())
    info.labels = labels.where(~labels.isin(PLACEHOLDERS | {""}))
    if clean.empty:
        info.semantic = "empty (placeholders only)"
        return info

    uniques = pd.Series(clean.unique())
    as_numbers = pd.to_numeric(uniques.str.replace(r"[,\s%₹$]", "", regex=True), errors="coerce")
    number_map = dict(zip(uniques, as_numbers))
    numeric_rate = clean.map(number_map).notna().mean()

    lower_uniques = set(uniques.str.lower())
    if lower_uniques <= BOOL_WORDS and len(lower_uniques) <= 2:
        info.semantic = "boolean (text)"
        return info

    if numeric_rate >= 0.95:
        info.semantic = "quantitative (stored as text)"
        info.numeric = series.astype(str).str.strip().map(number_map).astype(float)
        return info
    if 0.2 <= numeric_rate < 0.95:
        info.semantic = "mixed (numbers and text)"
        info.numeric = series.astype(str).str.strip().map(number_map).astype(float)
        return info

    has_digit = uniques.str.contains(r"\d", regex=True)
    if has_digit.mean() >= 0.9:
        order = _day_month_order(clean)
        dates = _parse_dates(series.astype("object").where(series.notna()).astype(str).str.strip()
                             .where(~(series.isna())), dayfirst=(order == "dayfirst"))
        date_rate = dates[clean.index].notna().mean()
        if date_rate >= 0.9 or (words & DATE_WORDS and date_rate >= 0.5):
            info.semantic = "datetime (stored as text)"
            info.dates = dates
            info.day_month = order
            return info

    unique_ratio = info.unique / len(present)
    if unique_ratio >= 0.95 and len(present) >= 20:
        no_spaces = (~uniques.str.contains(" ")).mean() >= 0.9
        info.semantic = "identifier" if (words & ID_WORDS or no_spaces) else "free text"
        return info
    if words & GEO_WORDS:
        info.semantic = "geographic (place names)"
        return info
    if info.unique <= 50 or unique_ratio <= 0.2:
        info.semantic = "categorical"
        return info
    info.semantic = "free text" if uniques.str.len().mean() > 30 else "categorical (many distinct values)"
    return info


def _is_quantitative(info: ColumnInfo) -> bool:
    return info.semantic in ("quantitative", "quantitative (stored as text)") and info.numeric is not None


def _is_category(info: ColumnInfo) -> bool:
    return info.semantic in ("categorical", "geographic (place names)", "categorical (many distinct values)",
                             "boolean (text)")


# ---------------------------------------------------------------------------
# Step 2-3: checks
# ---------------------------------------------------------------------------

def _fmt_n(x) -> str:
    return f"{int(x):,}"


def _list(items, limit=5) -> str:
    items = list(items)
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


class _Audit:
    def __init__(self, df: pd.DataFrame, today: dt.date):
        self.df = df
        self.n = len(df)
        self.today = today
        self.findings = []
        self.done, self.skipped = [], []
        self.info = {str(c): _profile_column(c, df[c], len(df)) for c in df.columns}
        self.stats = pd.DataFrame()
        self.correlations = pd.DataFrame()
        self.entity = None
        self.date_col = None
        self.key_unique = False

    def add(self, *args, **kwargs):
        finding = Finding(*args, **kwargs)
        self.findings.append(finding)
        for col in finding.columns:
            if col in self.info:
                self.info[col].findings.append(finding)
        return finding

    # -- structure --------------------------------------------------------
    def structure(self):
        self.done.append("Structure: rows, columns, header names")
        if self.n == 0 or self.df.shape[1] == 0:
            self.add(CRITICAL, "The file has no data rows" if self.df.shape[1] else "The file has no columns",
                     "Nothing can be analysed.", 0, 0, "structure",
                     "Check that the right file (and sheet) was uploaded and that it has a header row.")
            return False
        names = [str(c) for c in self.df.columns]
        unnamed = [c for c in names if c.lower().startswith("unnamed:")]
        if unnamed:
            severity = CRITICAL if len(unnamed) > len(names) / 2 else WARNING
            self.add(severity, f"{len(unnamed)} column(s) have no header name",
                     "The header row may be missing, merged or not on the first line, so columns may be mislabelled.",
                     len(unnamed), len(names), "structure",
                     "Open the file and check the header row is the first row and every column has a name.",
                     tuple(unnamed))
        repeated = [c for c in names if re.search(r"\.\d+$", c) and re.sub(r"\.\d+$", "", c) in names]
        if repeated:
            self.add(WARNING, "Some column names appear more than once",
                     f"Columns {_list(repeated)} were renamed automatically, so it is unclear which is which.",
                     len(repeated), len(names), "structure", "Give every column a unique name in the source file.",
                     tuple(repeated))
        return True

    # -- missingness -------------------------------------------------------
    def missingness(self):
        self.done.append("Missing values, including placeholder text such as '-' or 'unknown'")
        total_cells = self.n * self.df.shape[1]
        missing_cells = sum(i.missing for i in self.info.values())
        if total_cells and missing_cells / total_cells >= 0.5:
            self.add(CRITICAL, "Most of the table is empty",
                     "With over half of all cells missing, results are unlikely to be representative.",
                     missing_cells, total_cells, "structure",
                     "Check whether the file was exported completely and whether empty columns can be removed.")
        minor = []
        for info in self.info.values():
            pct = 100 * info.missing / self.n
            if pct >= 100:
                self.add(WARNING, f"Column '{info.name}' is entirely empty", "It cannot be used in any analysis.",
                         info.missing, self.n, "structure", f"Remove '{info.name}' or find out why it is empty.",
                         (info.name,))
            elif pct > 20:
                self.add(WARNING, f"{pct:.1f}% of values are missing in '{info.name}'",
                         "Averages, totals and comparisons using this column rest on fewer rows and may be biased "
                         "if values are missing for a reason.",
                         info.missing, self.n, "anomaly",
                         f"Find out why '{info.name}' is often missing before relying on it; report how many rows "
                         "each result uses.", (info.name,))
            elif info.missing:
                minor.append(info)
        if minor:
            rows = int(self.df[[m.name for m in minor]].isna().any(axis=1).sum())
            self.add(OBSERVATION, "Some missing values in " + _list(
                f"'{m.name}' ({100 * m.missing / self.n:.1f}%)" for m in minor),
                "Small amounts of missing data; most calculations simply skip them.",
                rows, self.n, "anomaly", "", tuple(m.name for m in minor))
        holders = [i for i in self.info.values() if i.placeholders]
        if holders:
            mask = np.zeros(self.n, dtype=bool)
            for info in holders:
                text = self.df[info.name].astype("object").where(self.df[info.name].notna()).astype(str).str.strip()
                mask |= (text.str.lower().isin(PLACEHOLDERS) | (text == "")).fillna(False).to_numpy()
            self.add(WARNING, "Placeholder text used for missing values in " + _list(
                f"'{i.name}' ({i.placeholders:,})" for i in holders),
                "Values such as '-' or 'unknown' are read as text, not as missing, so they can be counted as a "
                "category or stop a column being treated as numbers.",
                int(mask.sum()), self.n, "structure",
                "Treat these placeholders as missing (for example with na_values=[...] when reading the file).",
                tuple(i.name for i in holders))

    # -- duplicates and keys -----------------------------------------------
    def duplicates(self):
        self.done.append("Exact duplicate rows and identifier uniqueness")
        dup_rows = int(self.df.duplicated().sum())
        if dup_rows:
            pct = 100 * dup_rows / self.n
            self.add(CRITICAL if pct > 10 else WARNING, f"{dup_rows:,} rows are exact copies of another row",
                     "Duplicated rows are counted more than once, inflating totals and shifting averages.",
                     dup_rows, self.n, "invalid" if pct > 10 else "anomaly",
                     "Confirm whether the copies are errors and, if so, remove them with drop_duplicates().")
        for info in self.info.values():
            if info.semantic != "identifier" and not (_words(info.name) & ID_WORDS and _is_category(info)):
                continue
            values = self.df[info.name].dropna()
            repeated = int(values.duplicated().sum())
            if info.semantic == "identifier" and info.missing:
                self.add(WARNING, f"Likely ID column '{info.name}' has missing values",
                         "Rows without an ID cannot be matched or tracked.", info.missing, self.n, "anomaly",
                         f"Fill in or investigate rows with no '{info.name}'.", (info.name,))
            if repeated and _words(info.name) & ID_WORDS and info.unique > self.n * 0.5:
                self.add(WARNING, f"Some values of '{info.name}' appear more than once",
                         "If each row should be a different record, repeated IDs suggest duplicates; this may "
                         "be fine if the same entity appears in several rows.",
                         repeated, len(values), "domain",
                         f"Confirm whether '{info.name}' should be unique per row.", (info.name,))

    # -- types ---------------------------------------------------------------
    def types(self):
        self.done.append("Data types: numbers or dates stored as text, mixed types, unreadable dates")
        as_text = [i for i in self.info.values() if i.semantic == "quantitative (stored as text)"]
        if as_text:
            rows = int(self.df[[i.name for i in as_text]].notna().any(axis=1).sum())
            self.add(WARNING, "Numbers stored as text in " + _list(f"'{i.name}'" for i in as_text),
                     "These columns cannot be summed or averaged until converted; text like '1,200' or '45%' "
                     "may also be read incorrectly.",
                     rows, self.n, "structure",
                     "Convert with pd.to_numeric() after removing commas, % signs or units.",
                     tuple(i.name for i in as_text))
        for info in self.info.values():
            if info.semantic == "mixed (numbers and text)":
                numbers = int(info.numeric.notna().sum())
                texts = int(self.df[info.name].notna().sum()) - numbers - info.placeholders
                self.add(WARNING, f"'{info.name}' mixes numbers and text",
                         f"{numbers:,} values look like numbers and {texts:,} do not, so the column can be "
                         "used neither as numbers nor reliably as categories.",
                         texts, int(self.df[info.name].notna().sum()), "structure",
                         f"Check the non-numeric entries in '{info.name}' and decide how to code them.",
                         (info.name,))
            if info.semantic == "datetime (stored as text)":
                present = self.df[info.name].notna() & info.labels.notna()
                bad = int((present & info.dates.isna()).sum())
                if bad:
                    self.add(WARNING, f"{bad:,} values in '{info.name}' could not be read as dates",
                             "These rows will drop out of anything done by date.", bad, int(present.sum()),
                             "invalid", f"Look at the unreadable entries in '{info.name}' and correct their format.",
                             (info.name,))
                order = getattr(info, "day_month", None)
                if order == "dayfirst":
                    self.add(WARNING, f"Dates in '{info.name}' are written day/month/year",
                             "Python reads dates like 05/02/2025 as month first by default, so days 1-12 would "
                             "be swapped with months and trends would be wrong.",
                             None, None, "domain",
                             f"Convert with pd.to_datetime(df['{info.name}'], dayfirst=True) or an explicit format.",
                             (info.name,))
                elif order == "ambiguous":
                    self.add(WARNING, f"Day/month order cannot be told apart in '{info.name}'",
                             "No value has a day above 12, so 05/02 could mean 5 February or 2 May.",
                             None, None, "domain",
                             f"Confirm with the data owner whether '{info.name}' is day/month or month/day.",
                             (info.name,))
                elif order == "mixed":
                    self.add(WARNING, f"'{info.name}' mixes day/month and month/day dates",
                             "Some dates must be day-first and others month-first, so at least some are wrong.",
                             None, None, "invalid", f"Standardise the date format in '{info.name}' at source.",
                             (info.name,))
                else:
                    self.add(OBSERVATION, f"Dates in '{info.name}' are stored as text",
                             "They need converting before sorting, grouping by month or plotting over time.",
                             None, None, "structure", f"Convert with pd.to_datetime(df['{info.name}']).",
                             (info.name,))

    # -- value rules ---------------------------------------------------------
    def values(self):
        self.done.append("Value rules: impossible coordinates, percentages, negative values, start/end order, "
                         "dates outside a plausible range")
        for info in self.info.values():
            words = _words(info.name)
            if info.numeric is not None and info.semantic != "identifier":
                values = info.numeric.dropna()
                if words & {"lat", "latitude"} and len(values):
                    bad = int((values.abs() > 90).sum())
                    if bad:
                        self.add(WARNING, f"{bad:,} latitudes in '{info.name}' are outside -90 to 90",
                                 "These locations are impossible and will be misplaced on maps.", bad, len(values),
                                 "invalid", f"Correct or remove the impossible values in '{info.name}'.", (info.name,))
                if words & {"lon", "lng", "longitude"} and len(values):
                    bad = int((values.abs() > 180).sum())
                    if bad:
                        self.add(WARNING, f"{bad:,} longitudes in '{info.name}' are outside -180 to 180",
                                 "These locations are impossible and will be misplaced on maps.", bad, len(values),
                                 "invalid", f"Correct or remove the impossible values in '{info.name}'.", (info.name,))
                if (words & PERCENT_WORDS or "%" in info.name) and len(values):
                    bad = int(((values < 0) | (values > 100)).sum())
                    if bad:
                        self.add(WARNING, f"{bad:,} values in '{info.name}' are outside 0-100",
                                 "The name suggests a percentage; values outside 0-100 would be impossible.",
                                 bad, len(values), "domain",
                                 f"Confirm what '{info.name}' measures and its valid range.", (info.name,))
                if (words & NON_NEGATIVE_WORDS or any(w.startswith("pm") for w in words)) and len(values):
                    bad = int((values < 0).sum())
                    if bad:
                        self.add(WARNING, f"{bad:,} negative values in '{info.name}'",
                                 "The name suggests a quantity that cannot be negative; these may be errors or "
                                 "codes for missing data (such as -999).", bad, len(values), "domain",
                                 f"Confirm whether negative values in '{info.name}' are valid.", (info.name,))
            if info.dates is not None and info.semantic != "datetime (year)":
                dates = pd.to_datetime(info.dates, errors="coerce").dropna()
                if len(dates):
                    future = int((dates.dt.date > self.today).sum())
                    early = int((dates.dt.year < 1900).sum())
                    if future:
                        self.add(OBSERVATION, f"{future:,} dates in '{info.name}' are in the future",
                                 "Possible for forecasts or plans; otherwise likely typing errors.", future,
                                 len(dates), "domain", f"Check the future dates in '{info.name}'.", (info.name,))
                    if early:
                        self.add(OBSERVATION, f"{early:,} dates in '{info.name}' are before 1900",
                                 "Often a sign of a default or mistyped date.", early, len(dates), "domain",
                                 f"Check the very early dates in '{info.name}'.", (info.name,))
        # Start/end pairs
        dated = {k: v for k, v in self.info.items() if v.dates is not None}
        for start in dated:
            ws = _words(start)
            if not ws & {"start", "from", "begin", "open"}:
                continue
            for end in dated:
                we = _words(end)
                if end != start and we & {"end", "to", "finish", "close"} and \
                        (ws - {"start", "from", "begin", "open"}) == (we - {"end", "to", "finish", "close"}):
                    a = pd.to_datetime(dated[start].dates, errors="coerce")
                    b = pd.to_datetime(dated[end].dates, errors="coerce")
                    both = a.notna() & b.notna()
                    bad = int((both & (a > b)).sum())
                    if bad:
                        self.add(WARNING, f"{bad:,} rows have '{start}' after '{end}'",
                                 "A start after its end is impossible; durations for these rows would be negative.",
                                 bad, int(both.sum()), "invalid",
                                 f"Correct rows where '{start}' is later than '{end}'.", (start, end))

    # -- categories ----------------------------------------------------------
    def categories(self):
        self.done.append("Category labels: spelling, capitalisation and spacing variants, rare categories")
        rare_cols = []
        for info in self.info.values():
            if not _is_category(info) or info.labels is None or info.sensitive:
                continue
            raw = self.df[info.name].astype("object").where(self.df[info.name].notna()).dropna().astype(str)
            raw = raw[~raw.str.strip().str.lower().isin(PLACEHOLDERS | {""})]
            norm = raw.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            pairs = pd.DataFrame({"norm": norm, "raw": raw}).value_counts()
            forms_per_label = pairs.groupby(level=0).size()
            variant_groups = {}
            affected = 0
            for label in forms_per_label[forms_per_label > 1].index:
                counts = pairs.loc[label].sort_values(ascending=False)
                variant_groups[label] = counts
                affected += int(counts.iloc[1:].sum())
            if variant_groups:
                examples = []
                for counts in list(variant_groups.values())[:2]:
                    examples.append(" / ".join(f"'{v}'" for v in counts.index[:3]))
                self.add(WARNING, f"'{info.name}' has {len(variant_groups)} label(s) written in more than one way",
                         f"{raw.nunique():,} written forms stand for {norm.nunique():,} labels (for example "
                         f"{'; '.join(examples)}), so groups would be split and counted separately.",
                         affected, len(raw), "invalid",
                         f"Standardise '{info.name}' (for example .str.strip().str.title()) before grouping.",
                         (info.name,))
            counts = norm.value_counts()
            if 3 <= len(counts) <= 50 and len(norm) >= 50:
                rare = counts[counts < 5]
                if len(rare):
                    rare_cols.append((info.name, len(rare)))
                if counts.iloc[0] / len(norm) >= 0.95:
                    self.add(OBSERVATION, f"'{info.name}' is dominated by one category",
                             f"{100 * counts.iloc[0] / len(norm):.1f}% of rows share one label, so comparisons "
                             "between categories rest on few rows.", int(len(norm) - counts.iloc[0]), len(norm),
                             "anomaly", "", (info.name,))
        if rare_cols:
            self.add(OBSERVATION, "Rare categories in " + _list(f"'{c}' ({k} rare)" for c, k in rare_cols),
                     "Categories with fewer than 5 rows give unreliable group results; they may also be typos.",
                     None, None, "anomaly", "", tuple(c for c, _ in rare_cols))

    # -- personal data -------------------------------------------------------
    def personal_data(self):
        self.done.append("Possible personal or identifying information (patterns and column names)")
        flagged = []
        for info in self.info.values():
            words = _words(info.name)
            hits = {}
            column = self.df[info.name]
            text = None
            if _is_text(column):
                text = column.dropna().astype(str).str.strip()
            elif pd.api.types.is_integer_dtype(column) or info.semantic == "identifier":
                # Phone or Aadhaar numbers are often stored as plain numbers.
                whole = column.dropna()
                whole = whole[np.mod(whole.astype(float), 1) == 0]
                text = whole.astype("int64").astype(str)
            if text is not None:
                if len(text):
                    for label, pattern in SENSITIVE_PATTERNS.items():
                        count = int(text.str.match(pattern).sum())
                        if count:
                            hits[label] = count
            name_hint = bool(words & SENSITIVE_WORDS) or ("name" in words and bool(words & PERSON_WORDS))
            if hits or name_hint:
                info.sensitive = True
                what = ", ".join(f"{count:,} {label}" for label, count in hits.items()) or "column name suggests it"
                flagged.append(info.name)
                affected = max(hits.values()) if hits else int(self.df[info.name].notna().sum())
                self.add(WARNING, f"'{info.name}' may contain personal information ({what})",
                         "Personal data must not leave approved systems; check consent and confidentiality "
                         "obligations before sharing results or the file.",
                         affected, self.n, "domain",
                         f"Remove or pseudonymise '{info.name}' if it is not needed for the analysis.",
                         (info.name,))
        return flagged

    # -- statistics ------------------------------------------------------------
    def statistics(self):
        quant = [i for i in self.info.values() if _is_quantitative(i)]
        if not quant:
            self.skipped.append("Descriptive statistics, outliers and skewness: no numeric measurement columns")
            return
        self.done.append("Descriptive statistics, outliers (IQR rule, 1.5 x IQR) and skewness for numeric columns")
        rows, outlier_mask, outlier_cols, skewed, constant = [], np.zeros(self.n, dtype=bool), [], [], []
        for info in quant:
            values = info.numeric.dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            skew = float(values.skew()) if len(values) > 2 else float("nan")
            flagged = 0
            if len(values) >= 20 and iqr > 0:
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (info.numeric < low) | (info.numeric > high)
                flagged = int(mask.sum())
                if flagged:
                    outlier_mask |= mask.fillna(False).to_numpy()
                    outlier_cols.append((info.name, f"'{info.name}' ({flagged:,}, {100 * flagged / len(values):.1f}%)"))
            if values.nunique() == 1:
                constant.append(info.name)
            elif not math.isnan(skew) and abs(skew) > 1:
                skewed.append(f"'{info.name}' ({skew:+.1f})")
            rows.append({
                "column": info.name, "count": len(values), "mean": values.mean(), "std": values.std(),
                "min": values.min(), "Q1": q1, "median": values.median(), "Q3": q3, "max": values.max(),
                "IQR": iqr, "skewness": skew, "outliers (IQR)": flagged,
            })
        self.stats = pd.DataFrame(rows).round(3)
        if outlier_cols:
            self.add(OBSERVATION, "Possible outliers (IQR rule) in " + _list(text for _, text in outlier_cols),
                     "Unusually high or low values are worth checking; they may be real events or entry errors. "
                     "They are not errors by definition.",
                     int(outlier_mask.sum()), self.n, "anomaly",
                     "Look at the most extreme values with someone who knows the data before excluding any.",
                     tuple(name for name, _ in outlier_cols))
        if skewed:
            self.add(OBSERVATION, "Strongly skewed distributions in " + _list(skewed),
                     "A few large values pull the mean up; the median is usually a better summary.",
                     None, None, "anomaly", "", ())
        if constant:
            self.add(OBSERVATION, "Columns with a single value: " + _list(f"'{c}'" for c in constant),
                     "A column that never changes carries no information for comparisons.", None, None,
                     "structure", "", tuple(constant))

    def correlation(self):
        quant = [i for i in self.info.values() if _is_quantitative(i) and i.numeric.nunique() > 1]
        if len(quant) < 2:
            self.skipped.append("Correlation analysis: fewer than two numeric measurement columns")
            return
        quant = sorted(quant, key=lambda i: i.numeric.isna().mean())[:15]
        self.done.append(f"Correlation (Pearson and Spearman) between {len(quant)} numeric columns, "
                         "pairs with at least 30 complete rows")
        rows = []
        for a_i, a in enumerate(quant):
            for b in quant[a_i + 1:]:
                pair = pd.concat([a.numeric, b.numeric], axis=1).dropna()
                if len(pair) < 30:
                    continue
                pearson = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                # Spearman = Pearson on ranks; pandas' method="spearman" needs scipy, which the browser lacks.
                ranks = pair.rank()
                spearman = ranks.iloc[:, 0].corr(ranks.iloc[:, 1])
                if not np.isnan(pearson):
                    rows.append({"column A": a.name, "column B": b.name, "rows used": len(pair),
                                 "Pearson r": round(pearson, 3), "Spearman rho": round(spearman, 3)})
        if not rows:
            return
        table = pd.DataFrame(rows)
        table = table.reindex(table["Pearson r"].abs().sort_values(ascending=False).index).head(10)
        self.correlations = table.reset_index(drop=True)
        strong = table[table["Pearson r"].abs() >= 0.9]
        if len(strong):
            self.add(OBSERVATION, "Very strong correlations: " + _list(
                f"'{r['column A']}' and '{r['column B']}' (r = {r['Pearson r']:+.2f})" for _, r in strong.iterrows()),
                "Columns this closely related may measure the same thing or be derived from each other; using "
                "both in one model can mislead. Correlation does not show cause.", None, None, "anomaly", "", ())

    # -- time ------------------------------------------------------------------
    def temporal(self):
        dated = [i for i in self.info.values() if i.dates is not None and i.semantic != "datetime (year)"]
        if not dated:
            self.skipped.append("Time checks: no date or time column found")
            return
        dated.sort(key=lambda i: (not (_words(i.name) & DATE_WORDS), i.dates.isna().mean()))
        date_info = dated[0]
        self.date_col = date_info.name
        dates = pd.to_datetime(date_info.dates, errors="coerce")
        self.done.append(f"Time checks on '{date_info.name}': date range, repeated dates per entity, reporting gaps")

        # Is this a time series (about one row per entity per date)? Only then are repeated
        # entity+date rows suspicious and gaps in reporting meaningful.
        exact_copy = self.df.duplicated().to_numpy()

        def series_frame(entity):
            frame = pd.DataFrame({"date": dates})
            frame["entity"] = entity.labels.str.replace(r"\s+", " ", regex=True) if entity else "all"
            return frame

        def is_time_series(frame):
            usable = frame[~exact_copy].dropna()
            return len(usable) >= 10 and usable.groupby(["entity", "date"]).size().median() == 1

        candidates = [i for i in self.info.values() if _is_category(i) and i.labels is not None
                      and 2 <= i.labels.nunique() <= max(2, self.n // 2)]
        candidates.sort(key=lambda i: (not (_words(i.name) & ENTITY_WORDS), i.labels.nunique()))
        entity, frame = None, None
        for candidate in [None] + candidates:
            trial = series_frame(candidate)
            if is_time_series(trial):
                entity, frame = candidate, trial
                break
        if frame is None:
            self.skipped.append(f"Repeated-record and reporting-gap checks: '{date_info.name}' does not look like a "
                                "time series with one row per entity per date")
            return
        self.entity = entity.name if entity else None
        frame = frame.dropna()

        key = ["entity", "date"]
        exact_dup = self.df.duplicated(keep=False).to_numpy()
        key_dup = frame.duplicated(subset=key, keep=False).reindex(self.df.index, fill_value=False).to_numpy()
        conflicting = int((key_dup & ~exact_dup).sum())
        self.key_unique = conflicting == 0
        if entity and conflicting:
            pct = 100 * conflicting / self.n
            self.add(CRITICAL if pct > 10 else WARNING,
                     f"{conflicting:,} rows share the same '{entity.name}' and '{date_info.name}' but differ elsewhere",
                     "There is more than one, different record for the same entity and time, so it is unclear "
                     "which is correct; this may be fine if several readings per period are expected.",
                     conflicting, self.n, "domain",
                     f"Confirm whether each '{entity.name}' should have one row per '{date_info.name}'.",
                     (entity.name, date_info.name))

        gaps, usual = 0, None
        for _, group in frame.groupby("entity"):
            unique_dates = np.sort(group["date"].unique())
            if len(unique_dates) < 10:
                continue
            diffs = pd.Series(np.diff(unique_dates))
            median = diffs.median()
            if median <= pd.Timedelta(0):
                continue
            usual = median if usual is None else usual
            gaps += int((diffs > 2 * median).sum())
        if gaps:
            per = f" across '{entity.name}' groups" if entity else ""
            self.add(OBSERVATION, f"{gaps:,} gaps in reporting{per} longer than twice the usual interval "
                     f"({_fmt_interval(usual)})",
                     "Missing periods can bias averages over time and break moving averages or trend lines.",
                     gaps, None, "anomaly", "Check whether the gaps are real outages or missing records.",
                     (date_info.name,))

    # -- capabilities ----------------------------------------------------------
    def capabilities(self, flagged_personal):
        caps = []
        dup = next((f for f in self.findings if "exact copies" in f.issue), None)
        quant = [i for i in self.info.values() if _is_quantitative(i)]
        text_numbers = [i for i in quant if i.semantic == "quantitative (stored as text)"]
        cats = [i for i in self.info.values() if _is_category(i) and not i.sensitive]
        dated = [i for i in self.info.values() if i.dates is not None]
        ids = [i for i in self.info.values() if i.semantic == "identifier"]
        geo_xy = [i for i in self.info.values() if i.semantic == "geographic (coordinate)"]
        geo_names = [i for i in self.info.values() if i.semantic == "geographic (place names)"]

        def issues_for(cols):
            return [f for f in self.findings if f.severity in (CRITICAL, WARNING) and set(f.columns) & set(cols)]

        # Counts and totals
        if self.n == 0:
            caps.append(("Counting rows and totals", CANNOT, "The file has no rows."))
        elif dup:
            caps.append(("Counting rows and totals", CAREFUL, f"{dup.affected:,} duplicate rows would be counted twice."))
        else:
            caps.append(("Counting rows and totals", CAN, f"{self.n:,} rows, no exact duplicates."))

        names = _list(f"'{i.name}'" for i in quant)
        if not quant:
            caps.append(("Averages and summaries of measurements", CANNOT, "No numeric measurement columns."))
        else:
            problems = issues_for([i.name for i in quant])
            reasons = []
            if text_numbers:
                reasons.append("convert " + _list(f"'{i.name}'" for i in text_numbers) + " to numbers first")
            reasons += [p.issue[0].lower() + p.issue[1:] for p in problems if p not in [dup]][:2]
            if dup:
                reasons.append("remove duplicate rows first")
            caps.append(("Averages and summaries of " + names, CAREFUL if reasons else CAN,
                         _sentence("; ".join(reasons)) if reasons else "No problems found in these columns."))

        if not cats:
            caps.append(("Comparisons between groups", CANNOT, "No suitable category column."))
        else:
            problems = issues_for([i.name for i in cats])
            names = _list(f"'{i.name}'" for i in cats)
            caps.append(("Comparisons between groups (" + names + ")", CAREFUL if problems else CAN,
                         ("; ".join(p.issue for p in problems[:2]) + ".") if problems
                         else "Labels are consistent."))

        if not dated:
            caps.append(("Trends over time", CANNOT, "No date or time column."))
        else:
            reasons = []
            for info in dated:
                if info.semantic == "datetime (stored as text)":
                    reasons.append(f"'{info.name}' needs converting to dates")
            reasons += [f.issue for f in self.findings if f.severity in (CRITICAL, WARNING)
                        and set(f.columns) & {i.name for i in dated}]
            reasons += [f.issue for f in self.findings if "gaps in reporting" in f.issue]
            reasons = list(dict.fromkeys(reasons))[:3]
            caps.append(("Trends over time", CAREFUL if reasons else CAN,
                         ("; ".join(reasons) + ".") if reasons else "Dates are readable with no gaps found."))

        usable = [i for i in quant if i.numeric.notna().sum() >= 30]
        if len(usable) < 2:
            caps.append(("Relationships between measurements (correlation)", CANNOT,
                         "Needs at least two numeric columns with 30 or more values each."))
        else:
            caps.append(("Relationships between measurements (correlation)", CAN,
                         f"{len(usable)} numeric columns available. Correlation shows association, not cause."))

        if ids and not issues_for([i.name for i in ids]):
            caps.append(("Joining with other data / counting unique records", CAN,
                         f"'{ids[0].name}' uniquely identifies each row."))
        elif self.entity and self.date_col and self.key_unique:
            tidy = [step for step, needed in (
                ("removing duplicate rows", bool(dup)),
                ("tidying the labels", any("written in more than one way" in f.issue and self.entity in f.columns
                                           for f in self.findings))) if needed]
            status = CAREFUL if tidy else CAN
            caps.append(("Joining with other data / counting unique records", status,
                         f"'{self.entity}' + '{self.date_col}' together identify each row"
                         + (f" after {' and '.join(tidy)}." if tidy else ".")))
        elif ids:
            caps.append(("Joining with other data / counting unique records", CAREFUL,
                         f"'{ids[0].name}' has missing or repeated values."))
        else:
            caps.append(("Joining with other data / counting unique records", CANNOT,
                         "No column (or entity + date pair) uniquely identifies each row."))

        if geo_xy:
            caps.append(("Maps", CAREFUL if issues_for([i.name for i in geo_xy]) else CAN,
                         "Coordinates available: " + _list(f"'{i.name}'" for i in geo_xy) + "."))
        elif geo_names:
            caps.append(("Maps", CAREFUL, "Only place names (" + _list(f"'{i.name}'" for i in geo_names)
                         + "); they must be matched to a map boundary file or coordinates first."))
        else:
            caps.append(("Maps", CANNOT, "No coordinates or place-name columns."))

        if flagged_personal:
            caps.append(("Sharing the data outside your team", CAREFUL,
                         "Possible personal information in " + _list(f"'{c}'" for c in flagged_personal)
                         + "; check consent and confidentiality first."))
        else:
            caps.append(("Sharing the data outside your team", CAN,
                         "No personal-information patterns found (automatic checks cannot rule it out)."))
        return [{"Analysis": a, "Status": s, "Why": w} for a, s, w in caps]


def _sentence(text: str) -> str:
    """Capitalises only the first letter, so column names keep their case."""
    text = text.strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else text


def _fmt_interval(delta) -> str:
    if delta is None:
        return "unknown"
    seconds = delta.total_seconds()
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            value = seconds / size
            return f"{value:g} {unit}" + ("" if value == 1 else "s")
    return f"{seconds:g} seconds"


# ---------------------------------------------------------------------------
# Steps 5-8: verdicts, actions, report
# ---------------------------------------------------------------------------

def run_audit(df: pd.DataFrame, file_name: str = "data", today: dt.date | None = None) -> AuditReport:
    today = today or dt.date.today()
    # Work with text column names throughout (Excel headers can be numbers).
    df = df.set_axis([str(c) for c in df.columns], axis=1)
    audit = _Audit(df, today)
    ok = audit.structure()
    flagged = []
    if ok:
        audit.missingness()
        audit.duplicates()
        audit.types()
        flagged = audit.personal_data()
        audit.values()
        audit.categories()
        audit.statistics()
        audit.correlation()
        audit.temporal()

    order = {CRITICAL: 0, WARNING: 1, OBSERVATION: 2}
    findings = sorted(audit.findings, key=lambda f: (order[f.severity], -(f.pct or 0)))
    crit = [f for f in findings if f.severity == CRITICAL]
    warn = [f for f in findings if f.severity == WARNING]
    obs = [f for f in findings if f.severity == OBSERVATION]

    if crit:
        assessment = "Poor"
    elif warn:
        assessment = "Needs Attention"
    else:
        assessment = "Good"
    top = [f.issue for f in (crit + warn)[:2]]
    assessment_text = (
        f"The automatic checks found {len(crit)} critical issue(s), {len(warn)} warning(s) and "
        f"{len(obs)} observation(s) across {audit.n:,} rows and {df.shape[1]} columns. "
        + (f"The most important: {'; '.join(top)}. " if top else "No material problems were found. ")
        + "The checks are general and do not know your subject area, so unusual values need confirming "
          "by someone who knows the data."
    )

    if crit or not ok:
        readiness = NOT_READY
    elif warn:
        readiness = CAUTION
    else:
        readiness = READY
    first = (crit + warn)[:3]
    readiness_text = (
        "This verdict is provisional because the app does not know what you plan to analyse; the table "
        "'What you can analyse' shows readiness for common types of analysis. "
        + ("Resolve first: " + "; ".join(f.issue for f in first) + "." if first else "")
    )

    actions = []
    for f in crit + warn + obs:
        if f.action and f.action not in actions:
            actions.append(f.action)
    actions = actions[:7]

    capabilities = audit.capabilities(flagged) if ok else [
        {"Analysis": "Any analysis", "Status": CANNOT, "Why": "The file has no usable rows or columns."}]

    identifiers = [i.name for i in audit.info.values() if i.semantic == "identifier"]
    temporal = [i.name for i in audit.info.values() if i.dates is not None]
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return AuditReport(
        file_name=file_name, file_type=(file_name.rsplit(".", 1)[-1].upper() if "." in file_name else "unknown"),
        rows=audit.n, columns=df.shape[1], column_info=list(audit.info.values()), findings=findings,
        assessment=assessment, assessment_text=assessment_text, readiness=readiness,
        readiness_text=readiness_text, actions=actions, capabilities=capabilities, stats=audit.stats,
        correlations=audit.correlations, checks_done=audit.done, checks_skipped=audit.skipped,
        generated_at=generated, identifiers=identifiers, temporal=temporal,
    )


def column_summary(report: AuditReport) -> pd.DataFrame:
    rows = []
    severity_rank = {CRITICAL: 0, WARNING: 1, OBSERVATION: 2}
    for info in report.column_info:
        main = sorted(info.findings, key=lambda f: severity_rank[f.severity])
        rows.append({
            "Column": info.name,
            "Type": info.semantic + (" (possible personal data)" if info.sensitive else ""),
            "Missing %": round(100 * info.missing / report.rows, 1) if report.rows else 0.0,
            "Unique values": info.unique,
            "Main finding": f"{main[0].severity}: {main[0].issue}" if main else "No issues found",
        })
    return pd.DataFrame(rows)


def findings_table(findings) -> pd.DataFrame:
    return pd.DataFrame([{
        "Issue": f.issue,
        "Affected records": f"{f.affected:,}" if f.affected is not None else "-",
        "%": f"{f.pct:.1f}" if f.pct is not None else "-",
        "Impact": f.impact,
    } for f in findings])


# ---------------------------------------------------------------------------
# Downloadable report (HTML; use the browser's Print > Save as PDF for a PDF)
# ---------------------------------------------------------------------------

def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><em>None.</em></p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for value in row:
            if isinstance(value, float):
                value = "" if math.isnan(value) else f"{value:,.3f}".rstrip("0").rstrip(".")
            cells += f"<td>{html.escape(str(value))}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html_report(report: AuditReport) -> str:
    esc = html.escape
    crit, warn, obs = (report.by_severity(s) for s in (CRITICAL, WARNING, OBSERVATION))
    overview = pd.DataFrame([
        {"Item": "File", "Value": report.file_name},
        {"Item": "File type", "Value": report.file_type},
        {"Item": "Rows", "Value": f"{report.rows:,}"},
        {"Item": "Columns", "Value": f"{report.columns:,}"},
        {"Item": "Likely identifiers", "Value": ", ".join(report.identifiers) or "None found"},
        {"Item": "Date/time columns", "Value": ", ".join(report.temporal) or "None found"},
        {"Item": "Scope", "Value": "Full dataset, no sampling. Calculated in the browser; nothing was sent "
                                   "to any external service."},
    ])
    obs_html = "".join(
        f"<li><strong>{esc(f.issue)}</strong>"
        + (f" ({f.affected:,} records, {f.pct:.1f}%)" if f.pct is not None else "")
        + f". {esc(f.impact)}</li>" for f in obs) or "<li>No observations.</li>"
    actions_html = "".join(f"<li>{esc(a)}</li>" for a in report.actions) or "<li>No actions needed.</li>"
    done = "".join(f"<li>{esc(c)}</li>" for c in report.checks_done)
    skipped = "".join(f"<li>{esc(c)}</li>" for c in report.checks_skipped) or "<li>None.</li>"
    return f"""<!DOCTYPE html>
<html lang="en-GB"><head><meta charset="UTF-8"><title>Data Quality Audit Report - {esc(report.file_name)}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 2em auto; max-width: 960px; color: #1a1a1a;
       line-height: 1.45; padding: 0 1em; }}
h1 {{ font-size: 1.6em; margin-bottom: 0.2em; }} h2 {{ font-size: 1.2em; margin-top: 1.6em;
     border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 0.6em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ background: #f2f2f2; }} thead {{ display: table-header-group; }} tr {{ page-break-inside: avoid; }}
.verdict {{ font-size: 1.1em; padding: 0.6em 0.8em; background: #f6f6f6; border-left: 4px solid #888; }}
.meta {{ color: #555; font-size: 0.9em; }}
</style></head><body>
<h1>Data Quality Audit Report</h1>
<p class="meta">Dataset: {esc(report.file_name)}<br>Generated: {esc(report.generated_at)}<br>
Tabular Data Analyser, automatic checks following the data-quality-auditor-v2 method.</p>
<p class="verdict">Overall assessment: <strong>{esc(report.assessment)}</strong> &nbsp;|&nbsp;
Analysis readiness: <strong>{esc(report.readiness)}</strong> &nbsp;|&nbsp;
{len(crit)} critical, {len(warn)} warnings, {len(obs)} observations</p>
<h2>1. Dataset overview</h2>{_table(overview)}
<h2>2. Overall assessment: {esc(report.assessment)}</h2><p>{esc(report.assessment_text)}</p>
<h2>3. Critical issues</h2>{_table(findings_table(crit)) if crit else
    "<p>No critical issues identified by the checks performed.</p>"}
<h2>4. Warnings</h2>{_table(findings_table(warn)) if warn else
    "<p>No warnings identified by the checks performed.</p>"}
<h2>5. Observations</h2><ul>{obs_html}</ul>
<h2>6. Column-level quality summary</h2>{_table(column_summary(report))}
<h2>7. Recommended actions</h2><ol>{actions_html}</ol>
<h2>8. Analysis readiness: {esc(report.readiness)}</h2><p>{esc(report.readiness_text)}</p>
<h3>What you can analyse</h3>{_table(pd.DataFrame(report.capabilities))}
<h2>Statistical summary</h2>{_table(report.stats)}
<h3>Strongest correlations</h3>{_table(report.correlations)}
<h2>Methodology and limitations</h2>
<p>Checks performed:</p><ul>{done}</ul><p>Checks skipped:</p><ul>{skipped}</ul>
<p>Standard missing-value markers (empty cells, NA, N/A, null, NaN) were already read as missing when the file
was loaded. Outliers use the 1.5 x IQR rule and are candidates for checking, not errors. Rules based on column
names (for example percentages or values that cannot be negative) are flagged for confirmation, not treated as
proof. Values from columns that may contain personal information are never shown. The checks are general and
cannot know your subject area or how the data was collected.</p>
</body></html>
"""
