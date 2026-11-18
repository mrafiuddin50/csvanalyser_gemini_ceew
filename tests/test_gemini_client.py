"""Gemini client: key handling, model list, errors and code extraction."""
import io
import json
import sys
import urllib.error
import urllib.request
from unittest import mock

import pytest

from csvanalyser import gemini

# Gemini client

class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode() if not isinstance(body, bytes) else body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gemini_reply(text=None, **extra):
    body = {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]} if text is not None else {}
    body.update(extra)
    return body


def test_key_sent_in_header_not_url():
    seen = []

    def fake(req, *a, **k):
        seen.append(req)
        return FakeResponse(gemini_reply("```python\nresult = 1\n```"))
    with mock.patch("urllib.request.urlopen", fake):
        assert gemini.generate_python_code("p", "gemini-x", "  my-key  ") == "result = 1"
    assert "my-key" not in seen[0].full_url
    assert seen[0].get_header("X-goog-api-key") == "my-key"
    assert seen[0].full_url.endswith("/models/gemini-x:generateContent")


@pytest.mark.parametrize("reply, expected", [
    ("```python\nresult = 1\n```", "result = 1"),
    ("```py\nresult = 2\n```", "result = 2"),
    ("```\nresult = 3\n```", "result = 3"),
    ("Here you go:\n```python\nresult = 4\n```\nThanks", "result = 4"),
    ("```json\n{}\n```\n```python\nresult = 5\n```", "result = 5"),
    ("```python\nresult = 6\n", "result = 6"),       # unterminated
    ("result = 7", "result = 7"),                     # no fence
])
def test_extract_code(reply, expected):
    assert gemini.extract_code(reply) == expected


def _http_error(status, body):
    return urllib.error.HTTPError("u", status, "x", {}, io.BytesIO(json.dumps(body).encode()))


@pytest.mark.parametrize("status, body, expected", [
    (400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                     "details": [{"reason": "API_KEY_INVALID"}]}}, "rejected the API key"),
    (403, {"error": {"message": "Permission denied"}}, "not allowed to use the Gemini API"),
    (404, {"error": {"message": "models/x is not found"}}, "model is not available"),
    (429, {"error": {"message": "Resource exhausted"}}, "quota or rate limit"),
    (503, {"error": {"message": "overloaded"}}, "having problems right now"),
    (418, {"error": {"message": "teapot"}}, "error 418: teapot"),
])
def test_api_errors_raised_and_returned(status, body, expected):
    # Raised as HTTPError (CPython) ...
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(status, body)):
        with pytest.raises(gemini.GeminiError, match=expected) as info:
            gemini.generate_python_code("p", "m", "k")
    assert "Request failed" not in str(info.value) and "HTTP Error" not in str(info.value)
    # ... or returned with a non-200 status (Pyodide)
    with mock.patch("urllib.request.urlopen", return_value=FakeResponse(body, status)):
        with pytest.raises(gemini.GeminiError, match=expected):
            gemini.generate_python_code("p", "m", "k")


@pytest.mark.parametrize("body, expected", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked the request"),
    ({}, "empty response"),
    ({"candidates": [{"finishReason": "SAFETY"}]}, "finish reason: SAFETY"),
    ({"candidates": [{"content": {"parts": [{"text": "  "}]}, "finishReason": "MAX_TOKENS"}]}, "MAX_TOKENS"),
])
def test_blocked_and_empty_responses(body, expected):
    with mock.patch("urllib.request.urlopen", return_value=FakeResponse(body)):
        with pytest.raises(gemini.GeminiError, match=expected):
            gemini.generate_python_code("p", "m", "k")


def test_unreachable_network():
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
        with pytest.raises(gemini.GeminiError, match="Could not reach Gemini"):
            gemini.generate_python_code("p", "m", "k")


@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    urllib.error.URLError(TimeoutError("timed out")),
    urllib.error.URLError("The read operation timed out"),
])
def test_slow_replies_end_with_a_clear_message(error):
    with mock.patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(gemini.GeminiError, match="did not reply within 120 seconds"):
            gemini.generate_python_code("p", "m", "k")


def test_time_limits_are_passed_on():
    seen = []

    def fake(req, *a, timeout=None, **k):
        seen.append(timeout)
        if req.get_method() == "GET":
            return FakeResponse({"models": [{"name": "models/m", "supportedGenerationMethods": ["generateContent"]}]})
        return FakeResponse(gemini_reply("result = 1"))
    with mock.patch("urllib.request.urlopen", fake):
        gemini.list_models("k")
        gemini.generate_python_code("p", "m", "k")
        gemini.call_gemini_text("p", "m", "k")
    assert seen == [gemini.KEY_CHECK_TIMEOUT, gemini.ANSWER_TIMEOUT, gemini.ANSWER_TIMEOUT]


class FakeXHR:
    """Stands in for the browser's XMLHttpRequest (used when running inside stlite)."""
    made = []
    reply = (200, json.dumps(gemini_reply("```python\nresult = 1\n```")))
    error = None

    @classmethod
    def new(cls):
        xhr = cls()
        xhr.headers, xhr.sent, xhr.timeout = {}, None, 0
        cls.made.append(xhr)
        return xhr

    def open(self, method, url, is_async):
        self.method, self.url, self.is_async = method, url, is_async

    def setRequestHeader(self, name, value):
        self.headers[name] = value

    def send(self, body):
        self.sent = body
        if FakeXHR.error:
            raise FakeXHR.error
        self.status, self.responseText = FakeXHR.reply


class FakeJsError(Exception):
    """Like Pyodide's JsException for a DOMException named TimeoutError."""
    name = "TimeoutError"


def _in_browser():
    FakeXHR.made, FakeXHR.error = [], None
    fake_js = type(sys)("js")
    fake_js.XMLHttpRequest = FakeXHR
    return mock.patch.multiple(gemini, IN_BROWSER=True), mock.patch.dict(sys.modules, {"js": fake_js})


def test_browser_route_sets_time_limit_and_header():
    browser, js = _in_browser()
    with browser, js, mock.patch("urllib.request.urlopen", side_effect=AssertionError("urllib used")):
        assert gemini.generate_python_code("prompt", "gemini-x", " my-key ") == "result = 1"
    xhr = FakeXHR.made[0]
    assert (xhr.method, xhr.is_async, xhr.timeout) == ("POST", False, gemini.ANSWER_TIMEOUT * 1000)
    assert xhr.url.endswith("/models/gemini-x:generateContent") and "my-key" not in xhr.url
    assert xhr.headers["x-goog-api-key"] == "my-key"
    assert json.loads(xhr.sent)["contents"][0]["parts"][0]["text"] == "prompt"


def test_browser_route_time_limit_and_errors():
    browser, js = _in_browser()
    with browser, js:
        FakeXHR.error = FakeJsError("Failed to execute 'send' on 'XMLHttpRequest'")
        with pytest.raises(gemini.GeminiError, match="did not reply within 30 seconds"):
            gemini.list_models("k")
        assert FakeXHR.made[0].timeout == gemini.KEY_CHECK_TIMEOUT * 1000
        FakeXHR.error, FakeXHR.reply = None, (0, "")
        with pytest.raises(gemini.GeminiError, match="Could not reach Gemini"):
            gemini.generate_python_code("p", "m", "k")
        FakeXHR.reply = (429, json.dumps({"error": {"message": "Resource exhausted"}}))
        with pytest.raises(gemini.GeminiError, match="quota or rate limit"):
            gemini.generate_python_code("p", "m", "k")


def test_list_models_filters_orders_and_pages():
    pages = [
        {"models": [
            {"name": "models/zz-model", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
        ], "nextPageToken": "abc"},
        {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/imagen-4", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/aa-model", "supportedGenerationMethods": ["generateContent"]},
        ]},
    ]
    urls = []

    def fake(req, *a, **k):
        urls.append(req.full_url)
        return FakeResponse(pages[len(urls) - 1])
    with mock.patch("urllib.request.urlopen", fake):
        models = gemini.list_models("k")
    assert models == ["gemini-2.5-flash", "gemini-2.5-pro", "aa-model", "zz-model"]
    assert "pageToken=abc" in urls[1]
