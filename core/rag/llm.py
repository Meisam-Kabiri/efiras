"""Thin LLM wrapper — single call point for all pipeline layers."""

import json
import os
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("GPT_API_KEY")
        if not api_key:
            raise ValueError("GPT_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> Any:
    """
    Call an LLM and return either a parsed JSON dict (when schema is given)
    or a plain string.

    Args:
        model: Model id string (e.g. "gpt-4o-mini").
        messages: Chat messages in OpenAI format.
        schema: When set, forces JSON output and parses the response.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Parsed dict when schema is provided, else raw string content.
    """
    client = _get_openai_client()

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""

    if schema is not None:
        return json.loads(content)
    return content
