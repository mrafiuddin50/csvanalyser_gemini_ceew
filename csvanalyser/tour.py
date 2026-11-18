"""Steps of the in-app tour (opened with the 'Take the tour' button)."""

TOUR_STEPS = [
    (
        "Welcome",
        "**Tabular Data Analyser** lets you ask questions about a CSV or Excel file in plain English.\n\n"
        "- Python runs **inside your browser**, so your file never leaves your computer.\n"
        "- Google's Gemini writes the analysis code, but it only sees your question and the **column "
        "names and types**, never the values.\n\n"
        "This tour takes about two minutes. You can close it at any time.",
    ),
    (
        "1 · Load your data",
        "In the sidebar under **Data**, upload a CSV, TSV, TXT or Excel (.xlsx) file. The app works out "
        "the encoding and separator for you.\n\n"
        "No file to hand? Use **Try the synthetic sample**: made-up air-quality readings with a few "
        "problems built in to practise on.",
    ),
    (
        "2 · Check your data quality",
        "As soon as a file loads, the app audits it in your browser and shows a verdict banner:\n\n"
        "- **Data quality:** Good, Needs Attention or Poor.\n"
        "- **Analysis readiness:** Ready, Ready with caution or Not ready.\n\n"
        "The **① Data & quality** tab explains every finding and has a **What you can analyse** table "
        "showing which kinds of analysis are ready, need care or are not possible, and why. You can "
        "download the audit as a report.",
    ),
    (
        "3 · Connect to Gemini",
        "In the sidebar under **Connect**, paste your Gemini API key (free from Google AI Studio) and "
        "press Enter. The app checks it and lists the models your key can use.\n\n"
        "With a **free-tier** key, Google may use what is sent to improve its products; see "
        "*Help → How your data is handled*.",
    ),
    (
        "4 · Ask a question",
        "Open the **② Ask questions** tab. Type a question or click one of the **example questions**, "
        "then press **Analyse**.\n\n"
        "- **What will be sent to Gemini** shows the exact text before anything leaves your browser.\n"
        "- A progress box shows each step. If the code fails, Gemini is asked to fix it automatically "
        "(up to three attempts).\n"
        "- Follow-up questions build on your earlier questions.",
    ),
    (
        "5 · Learn from the answer",
        "Each answer is a card with tabs:\n\n"
        "- **Result:** the table, chart or number, with a CSV download.\n"
        "- **Code:** the Python behind it. Edit it and press **Run my version** (runs locally, nothing "
        "sent), or press **Explain this code** for a plain-language walkthrough.\n"
        "- **Sent to Gemini:** a record of every request for that answer.",
    ),
    (
        "6 · Take your work away",
        "The **③ Export** tab saves your session as a **Python script** or **Jupyter notebook**. They "
        "contain code only, so you can share them without sharing your data.\n\n"
        "The status line under the title and the **Activity log** in the sidebar show what has happened. "
        "For more detail, open the **printable guide**. Happy analysing!",
    ),
]
