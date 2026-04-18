"""LLM provider adapters for structured tool/function calling."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import yaml


def load_secrets(path: str = "secrets/secrets.yaml") -> dict:
    """Load API keys from a YAML file. Falls back gracefully if the file is missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


class LLMAdapter(ABC):
    """
    Provider-agnostic interface for LLM tool/function calling.

    Subclasses declare TOOL_FORMAT ("anthropic" or "openai") and implement
    _api_call(). The base class handles tool schema formatting.
    """

    TOOL_FORMAT: Literal["anthropic", "openai"]

    def _build_tools(self, tool_name: str, tool_desc: str, json_schema: dict) -> list:
        if self.TOOL_FORMAT == "anthropic":
            return [{"name": tool_name, "description": tool_desc, "input_schema": json_schema}]
        else:  # openai format — also accepted by Gemini FunctionDeclaration
            return [{"type": "function", "function": {
                "name": tool_name,
                "description": tool_desc,
                "parameters": json_schema,
            }}]

    def complete_with_tool(
        self,
        system: str,
        user: str,
        json_schema: dict,
        tool_name: str,
        tool_desc: str,
    ) -> dict:
        """Build the tool definition and delegate to the provider API call."""
        tools = self._build_tools(tool_name, tool_desc, json_schema)
        return self._api_call(system, user, tools, tool_name)

    @abstractmethod
    def _api_call(
        self, system: str, user: str, tools: list, tool_name: str
    ) -> dict:
        """Make the provider API call and return the raw parsed tool arguments."""
        ...


class AnthropicAdapter(LLMAdapter):
    TOOL_FORMAT = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        secrets_path: str = "secrets/secrets.yaml",
    ):
        import anthropic
        secrets = load_secrets(secrets_path)
        api_key = secrets.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _api_call(self, system: str, user: str, tools: list, tool_name: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return tool_block.input


class OpenAIAdapter(LLMAdapter):
    TOOL_FORMAT = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        secrets_path: str = "secrets/secrets.yaml",
    ):
        from openai import OpenAI
        secrets = load_secrets(secrets_path)
        api_key = secrets.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _api_call(self, system: str, user: str, tools: list, tool_name: str) -> dict:
        import json
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


class GeminiAdapter(LLMAdapter):
    TOOL_FORMAT = "openai"  # base class formats schema in openai style; we adapt below

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        secrets_path: str = "secrets/secrets.yaml",
    ):
        import google.generativeai as genai
        secrets = load_secrets(secrets_path)
        api_key = secrets.get("google_api_key") or os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def _api_call(self, system: str, user: str, tools: list, tool_name: str) -> dict:
        from google.generativeai.types import FunctionDeclaration, Tool
        fn_def = tools[0]["function"]  # already in openai format from base class
        fn = FunctionDeclaration(
            name=fn_def["name"],
            description=fn_def["description"],
            parameters=fn_def["parameters"],
        )
        mdl = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            tools=[Tool(function_declarations=[fn])],
        )
        fc = mdl.generate_content(user).candidates[0].content.parts[0].function_call
        return dict(fc.args)
