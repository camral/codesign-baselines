"""Strictly unwrap an optional JSON code fence from model completions."""

import re

_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)


def json_body(content):
    text = content.strip()
    if text.startswith("```"):
        match = _JSON_FENCE.fullmatch(text)
        return match.group(1).strip() if match else text
    return text
