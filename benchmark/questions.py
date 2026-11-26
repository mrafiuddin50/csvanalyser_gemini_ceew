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


@dataclass(frozen=True)
class ChartQuestion:
    """A chart request, scored on the CEEW data viz skill's rules.

    charts: chart kinds that fit (after the chart functions' own checks); empty means any chart, or none.
    highlight: the items the question names to highlight; nothing else may be highlighted.
    The reference chart comes from running `answer_code`; its numbers are compared with Gemini's.
    """
    id: str
    dataset: str
    question: str
    charts: tuple
    answer_code: str
    highlight: tuple = ()
    rel_tol: float = 0.01
    params: tuple = ()          # (name, value) pairs the chart must have, e.g. (("orientation", "vertical"),)


_STATION = "df['station'].str.strip().str.title()"
_FUEL = "df['cooking_fuel'].str.strip().str.upper().replace({'FIREWOOD': 'Firewood', 'KEROSENE': 'Kerosene', 'ELECTRIC': 'Electric'})"

CHART_QUESTIONS = [
    ChartQuestion("ch01", "air_quality",
                  "Show the average PM2.5 for each station as a chart. Station names may differ in capital letters "
                  "and spacing; treat those as the same station.",
                  ("bar", "lollipop"),
                  f"t = df.assign(station={_STATION}).groupby('station', as_index=False)['pm25_ugm3'].mean()\n"
                  "ceew.bar(t, 'station', 'pm25_ugm3')\nresult = t"),
    ChartQuestion("ch02", "air_quality",
                  "How did the monthly average PM2.5 across all stations change through 2024? Show a chart.",
                  ("line", "bar"),
                  "d = df.assign(month=pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp())\n"
                  "t = d.groupby('month', as_index=False)['pm25_ugm3'].mean()\n"
                  "ceew.line(t, 'month', 'pm25_ugm3')\nresult = t"),
    ChartQuestion("ch03", "air_quality",
                  "Show the monthly average PM2.5 of each station through 2024 in one chart. Station names may "
                  "differ in capital letters and spacing; treat those as the same station.",
                  ("line",),
                  f"d = df.assign(station={_STATION}, month=pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp())\n"
                  "t = d.groupby(['month', 'station'], as_index=False)['pm25_ugm3'].mean()\n"
                  "ceew.line(t, 'month', 'pm25_ugm3', series='station')\nresult = t"),
    ChartQuestion("ch04", "air_quality", "Show how the PM2.5 readings are distributed, in bins of 10.",
                  ("histogram",),
                  "ceew.histogram(df, 'pm25_ugm3', bin_width=10)\nresult = df['pm25_ugm3'].describe()"),
    ChartQuestion("ch05", "air_quality", "Is wind speed related to PM2.5? Show it in a chart.",
                  ("scatter",),
                  "ceew.scatter(df, 'wind_speed_ms', 'pm25_ugm3')\nresult = df[['wind_speed_ms', 'pm25_ugm3']].corr()"),
    ChartQuestion("ch06", "air_quality",
                  "For a technical annex, compare the spread of PM2.5 readings between stations (median, quartiles "
                  "and outliers). Station names may differ in capital letters and spacing.",
                  ("boxplot",),
                  f"d = df.assign(station={_STATION})\nceew.boxplot(d, 'station', 'pm25_ugm3')\n"
                  "result = d.groupby('station')['pm25_ugm3'].describe()"),
    ChartQuestion("ch07", "air_quality",
                  "Show the average PM2.5 for every station and month of 2024 as a heatmap. Station names may "
                  "differ in capital letters and spacing.",
                  ("heatmap",),
                  f"d = df.assign(station={_STATION}, month=pd.to_datetime(df['date']).dt.month)\n"
                  "t = d.groupby(['station', 'month'], as_index=False)['pm25_ugm3'].mean()\n"
                  "ceew.heatmap(t, 'station', 'month', 'pm25_ugm3')\nresult = t"),
    ChartQuestion("ch08", "household_survey",
                  "Show the share of households by main cooking fuel as a pie chart. Fuel names may differ in "
                  "capital letters and spacing.",
                  ("pie",),
                  f"t = ({_FUEL}).value_counts(normalize=True).mul(100).rename_axis('fuel').reset_index(name='share')\n"
                  "ceew.pie(t, 'fuel', 'share')\nresult = t"),
    ChartQuestion("ch09", "household_survey",
                  "Within each income band, what share of households uses each main cooking fuel? Show the shares "
                  "as parts of a whole. Fuel names may differ in capital letters and spacing.",
                  ("stacked_bar",),
                  f"t = pd.crosstab(df['income_band'], {_FUEL}, normalize='index').mul(100).reset_index()\n"
                  "ceew.stacked_bar(t, 'income_band', [c for c in t.columns if c != 'income_band'])\nresult = t"),
    ChartQuestion("ch10", "household_survey",
                  "Compare the average monthly electricity use (monthly_kwh) of each income band across the "
                  "districts in one chart.",
                  ("grouped_bar", "heatmap"),
                  "t = df.pivot_table(index='district', columns='income_band', values='monthly_kwh', aggfunc='mean')"
                  ".reset_index()\nt.columns.name = None\n"
                  "ceew.grouped_bar(t, 'district', ['Low', 'Middle', 'High'])\nresult = t"),
    ChartQuestion("ch11", "household_survey",
                  "Show how far each district's average household size is above or below the average across all "
                  "households.",
                  ("diverging_bar",),
                  "t = df.groupby('district', as_index=False)['household_size'].mean()\n"
                  "t['household_size'] = t['household_size'] - df['household_size'].mean()\n"
                  "ceew.diverging_bar(t, 'district', 'household_size', reference='the average')\nresult = t"),
    ChartQuestion("ch12", "household_survey",
                  "Show the percentage of households in each district whose main cooking fuel is LPG, and highlight "
                  "Lakeside. Fuel names may differ in capital letters and spacing.",
                  ("bar", "lollipop"),
                  "t = (df.assign(lpg=df['cooking_fuel'].str.strip().str.upper().eq('LPG'))"
                  ".groupby('district', as_index=False)['lpg'].mean())\nt['lpg'] = t['lpg'] * 100\n"
                  "ceew.bar(t, 'district', 'lpg', highlight=['Lakeside'])\nresult = t",
                  highlight=("Lakeside",)),
    # Requests the skill says no to: the chart must not carry an AI-written title, or a second axis.
    ChartQuestion("ch13", "household_survey",
                  "Make an eye-catching pie chart of households by income band with a bold title.",
                  ("pie",),
                  "t = df['income_band'].value_counts(normalize=True).mul(100).rename_axis('income_band')"
                  ".reset_index(name='share')\nceew.pie(t, 'income_band', 'share')\nresult = t"),
    ChartQuestion("ch14", "household_survey",
                  "Plot the average monthly_kwh and the average household_size by district on one chart with "
                  "two y-axes (a line and column chart).",
                  (),
                  "st.info('Line and column charts with two y-axes are made by the CEEW Outreach data viz team.')\n"
                  "t = df.groupby('district', as_index=False)['monthly_kwh'].mean()\n"
                  "ceew.bar(t, 'district', 'monthly_kwh')\nresult = t"),
    # Hourly monitor data, as exported from CPCB or OpenAQ: times stored as text, long monitor names.
    ChartQuestion("ch15", "hourly_air", "Give the average PM2.5 value by hour.",
                  ("line", "bar"),
                  "times = pd.to_datetime(df['observation_hour_start'])\n"
                  "t = df.groupby(times.dt.hour.rename('hour'))['average_value_in_hour'].mean().reset_index()\n"
                  "ceew.line(t, 'hour', 'average_value_in_hour')\nresult = t"),
    ChartQuestion("ch16", "hourly_air", "Plot a vertical box plot of PM2.5 by station.",
                  ("boxplot",),
                  "ceew.boxplot(df, 'location_name', 'average_value_in_hour', orientation='vertical')\n"
                  "result = df.groupby('location_name')['average_value_in_hour'].describe()",
                  params=(("orientation", "vertical"),)),
]
