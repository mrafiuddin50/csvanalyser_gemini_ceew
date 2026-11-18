"""One question end to end: prompt, generate, run locally, retry with sanitised errors."""

import pandas as pd

from .privacy import anonymise_text, make_aliases, restore_names, sanitise_error
from .prompts import build_prompt, describe_schema
from .sandbox import run_generated_code


MAX_ATTEMPTS = 3            # first try + up to two self-corrections


PAST_TURNS_IN_PROMPT = 3    # earlier questions + code sent for follow-ups


def prepare_request(df: pd.DataFrame, question: str, context: str, anonymise: bool):
    """Works out exactly what would be sent for a question."""
    aliases = make_aliases(df.columns) if anonymise else {}
    work_df = df.rename(columns=aliases) if aliases else df
    return {
        "aliases": aliases,
        "reverse": {alias: original for original, alias in aliases.items()},
        "work_df": work_df,
        "schema": describe_schema(work_df),
        "question": anonymise_text(question, aliases),
        "context": anonymise_text(context, aliases),
    }


def past_turns_for_prompt(history: list, anonymise: bool) -> list:
    """Earlier successful questions + code. Never includes results."""
    usable = [
        (turn["sent_question"], turn["code"])
        for turn in history
        # Code the user edited may contain values they typed, so it is never sent automatically.
        if turn.get("code") and not turn.get("error") and not turn.get("edited")
        and turn.get("anonymise") == anonymise
    ]
    return usable[-PAST_TURNS_IN_PROMPT:]


def analyse(question: str, df: pd.DataFrame, context: str, history: list, anonymise: bool, llm,
            progress=None) -> dict:
    """Asks the model for code, runs it locally and retries with sanitised errors.

    llm: callable taking the prompt text and returning Python code.
    progress: optional callable taking a short message, called at each step so the
        page can show what is happening.
    """
    report = progress or (lambda message: None)
    request = prepare_request(df, question, context, anonymise)
    past = past_turns_for_prompt(history, anonymise)
    turn = {
        "question": question,
        "sent_question": request["question"],
        "anonymise": anonymise,
        "sent_prompts": [],
        "code": "",
        "attempts": 0,
        "outputs": [],
        "result": None,
        "error": None,
    }

    failure = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(request["schema"], request["question"], request["context"], past, failure)
        turn["sent_prompts"].append(prompt)
        turn["attempts"] = attempt
        report(f"Asking Gemini to write the code (attempt {attempt} of {MAX_ATTEMPTS})…"
               if attempt == 1 else f"Asking Gemini to fix the code (attempt {attempt} of {MAX_ATTEMPTS})…")
        try:
            code = llm(prompt)
        except Exception as exc:
            turn["error"] = f"Gemini request failed: {exc}"
            report("Gemini request failed.")
            return turn
        turn["code"] = code
        report("Running the code on your computer…")
        try:
            result, outputs = run_generated_code(code, request["work_df"], request["reverse"])
        except Exception as exc:
            failure = (code, sanitise_error(exc, code, request["work_df"]))
            # Full error is shown only to the user, never sent to the model.
            turn["error"] = f"{type(exc).__name__}: {restore_names(str(exc), request['reverse'])}"
            report(f"The code did not work ({type(exc).__name__})."
                   + (" Retrying…" if attempt < MAX_ATTEMPTS else " No attempts left."))
            continue
        turn["result"], turn["outputs"], turn["error"] = result, outputs, None
        report("Done.")
        return turn
    return turn


def run_edited_code(base_turn: dict, code: str, df: pd.DataFrame) -> dict:
    """Runs code the user edited, locally only. Nothing is sent to Gemini."""
    request = prepare_request(df, "", "", base_turn.get("anonymise", False))
    turn = {
        "question": base_turn["question"],
        "sent_question": base_turn.get("sent_question", base_turn["question"]),
        "anonymise": base_turn.get("anonymise", False),
        "edited": True,
        "sent_prompts": [],
        "code": code,
        "attempts": 1,
        "outputs": [],
        "result": None,
        "error": None,
    }
    try:
        turn["result"], turn["outputs"] = run_generated_code(code, request["work_df"], request["reverse"])
    except Exception as exc:
        turn["error"] = f"{type(exc).__name__}: {restore_names(str(exc), request['reverse'])}"
    return turn
