"""
LLM-based enrichment for enhanced pattern detection and documentation
"""

from dataclasses import dataclass, field
from json import dumps, loads
from logging import debug, error, info
from os import getenv
from re import DOTALL, search
from typing import Any, Dict, List, Literal, Optional, Union

from anthropic import Anthropic
from anthropic.types import MessageParam
from ollama import Client
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam


@dataclass
class LLMEnricher:
    """
    Optional LLM-based enrichment for column analysis.
    Supports OpenAI, Anthropic, and Ollama for local LLMs.

    Args:
        provider: LLM provider to use ("openai", "anthropic", or "ollama")
        api_key: Optional API key for LLM service (not needed for Ollama)
        model: Optional model name override
        ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
    Returns:
        An instance of LLMEnricher with methods to analyze columns and generate documentation.
    """

    provider: Literal["openai", "anthropic", "ollama"] = "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    client: Optional[Union[OpenAI, Anthropic, Client]] = field(init=False)
    enabled: bool = field(init=False, default=False)

    def __post_init__(self):

        if not self.api_key:
            env_key = f"{self.provider.upper()}_API_KEY"
            debug(f"No API key provided, checking environment variables for {env_key}")
            if env_key:
                info(f"Using API key from environment variable {env_key}")

                self.api_key = getenv(env_key)

        # Set default models
        if not self.model:
            match self.provider:
                case "openai":
                    self.model = "gpt-4o-mini"
                case "anthropic":
                    self.model = "claude-3-haiku-20240307"
                case "ollama":
                    self.model = "gemma3:270m"

        try:
            if self.provider == "openai" and self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                self.enabled = True
            elif self.provider == "anthropic" and self.api_key:
                self.client = Anthropic(api_key=self.api_key)
                self.enabled = True
            elif self.provider == "ollama":
                try:
                    self.client = Client(host=self.ollama_base_url)
                    # Test connection
                    self.client.list()
                    self.enabled = True
                except Exception as e:
                    error(f"Ollama client error: {e}, falling back to httpx get")
                    # Fallback to requests-based implementation
                    from httpx import get

                    response = get(f"{self.ollama_base_url}/api/tags")
                    if response.status_code == 200:
                        self.enabled = True
        except ImportError as ie:
            error(f"LLM client import error: {ie}")
        except Exception as e:
            error(e)

    def analyze_column(
        self, column_name: str, sample_values: List[str], dtype: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze column and suggest improvements.

        Args:
            column_name: Name of the column
            sample_values: Sample values from the column
            dtype: Data type of the column
            properties: Existing properties detected

        Returns:
            Dict with LLM suggestions:
                - description: Human-readable description
                - semantic_type: Semantic meaning (e.g., "customer_id", "price")
                - suggested_validations: Additional validation rules
                - business_rules: Potential business rules
        """
        if not self.enabled:
            return {}

        # Prepare prompt for LLM
        prompt = self._create_analysis_prompt(column_name, sample_values, dtype, properties)

        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            # Fallback to heuristic-based analysis
            error(f"LLM analysis failed: {e}, falling back to heuristic.")
            return {
                "description": f"Column '{column_name}' of type {dtype}",
                "semantic_type": self._infer_semantic_type(column_name),
                "suggested_validations": [],
                "business_rules": [],
            }

    def _create_analysis_prompt(
        self, column_name: str, sample_values: List[str], dtype: str, properties: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM analysis."""
        prompt = f"""Analyze this data column and provide insights:

Column Name: {column_name}
Data Type: {dtype}
Sample Values: {', '.join(str(v) for v in sample_values[:10])}
Properties: {dumps(properties, indent=2)}

Please provide a JSON response with:
1. "description": A clear, concise description of what this column represents
2. "semantic_type": The semantic meaning (e.g., "customer_id", "email", "price", "timestamp")
3. "suggested_validations": List of additional validation rules that should be applied
4. "business_rules": List of potential business rules or constraints

Respond with valid JSON only."""
        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the appropriate LLM provider."""
        sys = "You are a data analyst expert. Provide JSON responses only."
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    ChatCompletionSystemMessageParam(content=sys),
                    ChatCompletionUserMessageParam(content=prompt),
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                messages=[MessageParam(role="user", content=prompt)],
                system=sys,
                temperature=0.1,
                max_tokens=500,
            )
            return str(response.content[0].text)

        elif self.provider == "ollama":
            if hasattr(self.client, "chat"):
                # Using ollama Python package
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=sys,
                    options={"temperature": 0.1},
                )
                return response["message"]["content"]
            else:
                # Using requests fallback
                from httpx import post

                response = post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a data analyst expert. Provide JSON responses only.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                return response.json()["message"]["content"]

        return "{}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data."""
        try:
            # Try to extract JSON from response

            json_match = search(r"\{.*\}", response, DOTALL)
            if json_match:
                return loads(json_match.group())
            return loads(response)
        except Exception as e:
            error(
                f"Failed to parse LLM response: {e}, response was: {response}. returning empty dict."
            )
            return {}

    def _infer_semantic_type(self, column_name: str) -> str:
        """
        Infer semantic type from column name.
        This is a simple heuristic that would be replaced by LLM in production.
        """
        name_lower = column_name.lower()

        # Common semantic patterns
        if any(x in name_lower for x in ["id", "identifier", "key"]):
            return "identifier"
        elif any(x in name_lower for x in ["name", "title"]):
            return "name"
        elif any(x in name_lower for x in ["email", "mail"]):
            return "email"
        elif any(x in name_lower for x in ["phone", "tel", "mobile"]):
            return "phone"
        elif any(x in name_lower for x in ["date", "time", "timestamp"]):
            return "temporal"
        elif any(x in name_lower for x in ["price", "cost", "amount", "total"]):
            return "monetary"
        elif any(x in name_lower for x in ["count", "quantity", "number"]):
            return "quantity"
        elif any(x in name_lower for x in ["url", "link", "website"]):
            return "url"
        elif any(x in name_lower for x in ["address", "street", "city", "zip"]):
            return "address"
        elif any(x in name_lower for x in ["description", "comment", "note"]):
            return "text"
        elif any(x in name_lower for x in ["status", "state", "type", "category"]):
            return "categorical"
        elif any(x in name_lower for x in ["flag", "is_", "has_", "enabled"]):
            return "boolean"
        else:
            return "unknown"

    def generate_documentation(self, model_name: str, columns: List[Dict[str, Any]]) -> str:
        """
        Generate documentation for the model.

        Args:
            model_name: Name of the model
            columns: List of column information dicts

        Returns:
            Markdown documentation string
        """
        doc = f"# {model_name} Schema Documentation\n\n"
        doc += "## Overview\n\n"
        doc += f"This schema defines the structure for {model_name} data.\n\n"
        doc += "## Columns\n\n"

        for col in columns:
            doc += f"### {col['name']}\n\n"
            doc += f"- **Type**: {col.get('type', 'Unknown')}\n"

            if col.get("nullable"):
                doc += "- **Nullable**: Yes\n"
            if col.get("unique"):
                doc += "- **Unique**: Yes\n"
            if col.get("pattern_name"):
                doc += f"- **Pattern**: {col['pattern_name']}\n"
            if col.get("min_value") is not None:
                doc += f"- **Min Value**: {col['min_value']}\n"
            if col.get("max_value") is not None:
                doc += f"- **Max Value**: {col['max_value']}\n"
            if col.get("examples"):
                doc += f"- **Examples**: {', '.join(str(e) for e in col['examples'][:3])}\n"

            doc += "\n"

        return doc
