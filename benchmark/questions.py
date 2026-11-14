"""Benchmark questions. Each reference answer comes from running `answer_code` on the dataset.

kind:
  number  compared within a tolerance (relative, with an absolute floor)
  label   compared as text, ignoring case and surrounding spaces
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    id: str
    dataset: str
    question: str
    kind: str
    answer_code: str
    rel_tol: float = 0.005
    abs_tol: float = 1e-6


QUESTIONS = [
    # Air-quality sample: five fictional stations, 2024.
    Question("aq01", "air_quality", "What is the average PM2.5 (pm25_ugm3) across all readings? Ignore missing values.",
             "number", "result = df['pm25_ugm3'].mean()"),
    Question("aq02", "air_quality", "How many rows remain after removing exact duplicate rows?",
             "number", "result = len(df.drop_duplicates())"),
    Question("aq03", "air_quality",
             "How many distinct stations are there once differences in capital letters and extra spaces in the "
             "station names are ignored?",
             "number", "result = df['station'].str.strip().str.lower().nunique()"),
    Question("aq04", "air_quality",
             "Which station has the highest average PM2.5? Station names may differ in capital letters and spacing; "
             "treat those as the same station. Answer with the station name.",
             "label", "result = df.groupby(df['station'].str.strip().str.lower())['pm25_ugm3'].mean().idxmax()"),
    Question("aq05", "air_quality",
             "In which month of 2024 was the average PM2.5 across all stations highest? "
             "Answer with the month number (1-12).",
             "number", "result = int(df.groupby(pd.to_datetime(df['date']).dt.month)['pm25_ugm3'].mean().idxmax())"),
    Question("aq06", "air_quality",
             "On how many distinct dates did Green Park have PM2.5 above 60? Station names may differ in capital "
             "letters and spacing.",
             "number",
             "g = df[df['station'].str.strip().str.lower() == 'green park']\n"
             "result = g.loc[g['pm25_ugm3'] > 60, 'date'].nunique()"),
    Question("aq07", "air_quality",
             "What is the Pearson correlation between wind speed and PM2.5, using rows where both are present?",
             "number", "result = df['wind_speed_ms'].corr(df['pm25_ugm3'])", abs_tol=0.005),
    Question("aq08", "air_quality",
             "What percentage of PM10 values are missing? Answer as a percentage between 0 and 100.",
             "number", "result = df['pm10_ugm3'].isna().mean() * 100"),
    Question("aq09", "air_quality", "What is the median temperature across all stations in July 2024?",
             "number", "result = df.loc[df['date'].str.startswith('2024-07'), 'temperature_c'].median()"),
    Question("aq10", "air_quality",
             "What is the mean of the row-by-row ratio PM10 / PM2.5, using only rows where both values are present?",
             "number",
             "d = df.dropna(subset=['pm10_ugm3', 'pm25_ugm3'])\nresult = (d['pm10_ugm3'] / d['pm25_ugm3']).mean()"),

    # Household survey: 600 fictional households.
    Question("sv01", "household_survey",
             "What percentage of households use LPG as their main cooking fuel? Fuel names may differ in capital "
             "letters and spacing. Answer as a percentage between 0 and 100.",
             "number", "result = (df['cooking_fuel'].str.strip().str.upper() == 'LPG').mean() * 100"),
    Question("sv02", "household_survey",
             "What is the average monthly electricity use (monthly_kwh) of households in the High income band? "
             "Ignore missing values.",
             "number", "result = df.loc[df['income_band'] == 'High', 'monthly_kwh'].mean()"),
    Question("sv03", "household_survey",
             "Which district has the largest average household size? Answer with the district name.",
             "label", "result = df.groupby('district')['household_size'].mean().idxmax()"),
    Question("sv04", "household_survey",
             "How many households have no value recorded for monthly electricity use?",
             "number", "result = int(df['monthly_kwh'].isna().sum())"),
    Question("sv05", "household_survey",
             "What is the median monthly electricity use per person (monthly_kwh divided by household_size), "
             "ignoring households with missing values?",
             "number", "result = (df['monthly_kwh'] / df['household_size']).median()"),
    Question("sv06", "household_survey",
             "How many households were surveyed in February 2025? Survey dates are written day/month/year.",
             "number", "result = int((pd.to_datetime(df['survey_date'], format='%d/%m/%Y').dt.month == 2).sum())"),
]
