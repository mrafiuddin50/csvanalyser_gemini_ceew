"""Running edited code locally and the 'Explain this code' prompt."""


import pandas as pd

from csvanalyser import analysis, prompts


def make_df():
    return pd.DataFrame({
        "site_name": ["Anand Vihar", "Punjabi Bagh", "Okhla Phase-2", "Anand Vihar"],
        "pm25": [123.45, 678.91, None, 123.45],
        "date": ["2023-01-05", "2023-02-17", "2023-03-09", "2023-01-05"],
    })


def fake_llm(code):
    return lambda prompt: code


# Edited code and Explain

def test_run_edited_code_runs_locally():
    base = analysis.analyse("q", make_df(), "", [], False, fake_llm("result = df['pm25'].max()"))
    edited = analysis.run_edited_code(base, "result = df['pm25'].min()", make_df())
    assert edited["edited"] and edited["result"] == 123.45 and edited["sent_prompts"] == []
    broken = analysis.run_edited_code(base, "result = df['nope']", make_df())
    assert "KeyError" in broken["error"]


def test_edited_turns_never_sent_as_follow_up_context():
    base = analysis.analyse("q", make_df(), "", [], False, fake_llm("result = 1"))
    edited = analysis.run_edited_code(base, "result = df[df.site_name == 'Anand Vihar']", make_df())
    past = analysis.past_turns_for_prompt([base, edited], False)
    assert past == [("q", "result = 1")]


def test_edited_anonymised_turn_uses_aliases():
    base = analysis.analyse("q", make_df(), "", [], True, fake_llm("result = df['col_2'].max()"))
    edited = analysis.run_edited_code(base, "result = df.groupby('col_1')['col_2'].max()", make_df())
    assert edited["error"] is None and edited["result"].index.name == "site_name"


def test_explain_prompt_has_only_code_and_schema():
    df = make_df()
    request = analysis.prepare_request(df, "", "", False)
    prompt = prompts.build_explain_prompt(request["schema"], "result = df['pm25'].mean()")
    assert "result = df['pm25'].mean()" in prompt and "'site_name'" in prompt
    for secret in ["Anand Vihar", "Punjabi Bagh", "123.45", "678.91", "2023-01-05"]:
        assert secret not in prompt
