"""Gemini REST API client. The API key is sent in a header, never in the URL."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


# Shown before a key is entered, or if the model list cannot be fetched.
# Also sets the preferred order when the real list is available.
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-pro-latest",
]


# Models that support generateContent but are not text/code models.
EXCLUDED_MODEL_WORDS = (
    "embedding", "aqa", "tts", "image", "imagen", "veo",
    "native-audio", "live", "robotics", "computer-use",
)


class GeminiError(Exception):
    """A Gemini API problem, with a message written for the user."""


def describe_api_error(status: int, body: str) -> str:
    """Turns an HTTP error from Gemini into a plain-language message."""
    message, reasons = "", set()
    try:
        error = json.loads(body).get("error", {})
        message = error.get("message", "")
        reasons = {d.get("reason") for d in error.get("details", []) if isinstance(d, dict)}
    except (ValueError, AttributeError):
        pass

    if "API_KEY_INVALID" in reasons or "API key not valid" in message:
        return "Gemini rejected the API key. Check it was copied correctly from Google AI Studio."
    if status == 403:
        return f"This API key is not allowed to use the Gemini API (permission denied). {message}".strip()
    if status == 404:
        return "The selected model is not available for this API key. Choose another model in the sidebar."
    if status == 429:
        return ("The quota or rate limit for this API key has been reached. "
                "Wait a minute and try again, or choose another model.")
    if status >= 500:
        return f"Gemini is having problems right now (error {status}). Try again shortly."
    return f"Gemini returned error {status}: {message or body[:300]}"


def _gemini_request(path: str, api_key: str, payload=None) -> dict:
    """Calls the Gemini REST API. The key goes in a header, never in the URL."""
    headers = {"x-goog-api-key": api_key.strip()}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{GEMINI_BASE_URL}/{path}", data=data, headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        # In Pyodide, urlopen may return 4xx/5xx responses instead of raising.
        with urllib.request.urlopen(req) as response:
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        reason = getattr(exc, "reason", exc)
        raise GeminiError(f"Could not reach Gemini ({reason}). Check your internet connection.") from None

    if status != 200:
        raise GeminiError(describe_api_error(status, body))
    try:
        return json.loads(body)
    except ValueError:
        raise GeminiError("Gemini sent a response that could not be read. Try again.") from None


def order_models(names) -> list:
    """Preferred models first (in FALLBACK_MODELS order), then the rest A-Z."""
    names = list(dict.fromkeys(names))
    preferred = [m for m in FALLBACK_MODELS if m in names]
    return preferred + sorted(m for m in names if m not in preferred)


def list_models(api_key: str) -> list:
    """Text models this key can use for generateContent."""
    models, page_token = [], None
    for _ in range(5):
        path = "models?pageSize=1000"
        if page_token:
            path += "&pageToken=" + urllib.parse.quote(page_token)
        data = _gemini_request(path, api_key)
        for model in data.get("models", []):
            if "generateContent" not in model.get("supportedGenerationMethods", []):
                continue
            name = model.get("name", "").removeprefix("models/")
            if name and not any(word in name for word in EXCLUDED_MODEL_WORDS):
                models.append(name)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return order_models(models)


def extract_code(text: str) -> str:
    """Pulls Python code out of a reply, whether or not the block is tagged."""
    blocks = re.findall(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", text, re.DOTALL)
    for language, body in blocks:
        if language.lower() in ("python", "py", "python3"):
            return body.strip()
    if blocks:
        return blocks[0][1].strip()
    unterminated = re.search(r"```[ \t]*(?:python3?|py)?[^\n]*\n(.*)", text, re.DOTALL)
    if unterminated:
        return unterminated.group(1).strip()
    return text.strip()


def generate_python_code(prompt: str, model_name: str, api_key: str) -> str:
    """Asks Gemini for code and returns just the code."""
    code = extract_code(call_gemini_text(prompt, model_name, api_key))
    if not code:
        raise GeminiError("Gemini's reply did not contain any code. Try again.")
    return code


def call_gemini_text(prompt: str, model_name: str, api_key: str) -> str:
    """Sends one prompt to Gemini and returns the reply text."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    data = _gemini_request(f"models/{model_name}:generateContent", api_key, payload)

    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        raise GeminiError(f"Gemini blocked the request ({block_reason}). Try rephrasing the question.")
    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiError("Gemini returned an empty response. Try again or choose another model.")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        reason = candidate.get("finishReason", "unknown")
        raise GeminiError(f"Gemini returned no text (finish reason: {reason}). Try rephrasing the question.")
    return text
