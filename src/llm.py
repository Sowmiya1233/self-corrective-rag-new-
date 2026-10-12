"""Thin Groq wrapper with retries and robust JSON parsing (supports gpt-oss reasoning models)."""
import json
import re
import time

from groq import Groq

import config

_client = None
calls = 0  # counts LLM calls (used by the evaluation to report cost)
_json_mode_ok = True


class DailyLimitError(RuntimeError):
    """Raised when Groq's tokens-per-day quota is used up (retrying will not help)."""


def _get_client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing. Copy .env.example to .env and add your key.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _is_reasoning_model():
    return "gpt-oss" in config.GROQ_MODEL.lower()


def chat(messages, temperature=0.0, max_tokens=700, json_mode=False):
    global calls, _json_mode_ok
    reasoning = _is_reasoning_model()
    if reasoning:
        # reasoning tokens count toward the limit, so leave room for the real answer
        max_tokens = max(max_tokens, 3000)

    last_err = None
    for attempt in range(5):
        try:
            kwargs = dict(model=config.GROQ_MODEL, messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            if json_mode and _json_mode_ok:
                kwargs["response_format"] = {"type": "json_object"}
            if reasoning:
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            resp = _get_client().chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            calls += 1
            if config.REQUEST_DELAY:
                time.sleep(config.REQUEST_DELAY)
            if not content:
                raise RuntimeError("empty response (reasoning used all tokens?)")
            return content
        except Exception as e:  # rate limit / transient errors
            last_err = e
            msg = str(e).lower()
            if "tokens per day" in msg or "(tpd)" in msg:
                raise DailyLimitError(str(e))
            if "model_not_found" in msg or "does not exist" in msg:
                break  # retrying will not help; fail fast with the real error
            if json_mode and ("response_format" in msg or "json" in msg):
                _json_mode_ok = False  # model rejected JSON mode; fall back to plain text + parsing
            time.sleep(2 * (2 ** attempt))
    raise RuntimeError(f"Groq request failed after retries: {last_err}")


def chat_json(messages, **kwargs):
    """Ask for JSON and parse it; returns {} if parsing fails."""
    text = chat(messages, json_mode=True, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}