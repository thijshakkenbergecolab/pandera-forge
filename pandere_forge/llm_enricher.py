"""
LLM-based enrichment for enhanced pattern detection and documentation
"""

from typing import Dict, List, Optional, Any, Literal
import pandas as pd
import json
import os


class LLMEnricher:
    """
    Optional LLM-based enrichment for column analysis.
    Supports OpenAI, Anthropic, and Ollama for local LLMs.
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "ollama"] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LLM enricher.

        Args:
            provider: LLM provider to use ("openai", "anthropic", or "ollama")
            api_key: Optional API key for LLM service (not needed for Ollama)
            model: Optional model name override
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.ollama_base_url = ollama_base_url

        # Set default models
        if model:
            self.model = model
        elif provider == "openai":
            self.model = "gpt-4o-mini"
        elif provider == "anthropic":
            self.model = "claude-3-haiku-20240307"
        elif provider == "ollama":
            self.model = "llama3.2"

        # Initialize provider client
        self.client = None
        self.enabled = False

        try:
            if provider == "openai" and self.api_key:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.enabled = True
            elif provider == "anthropic" and self.api_key:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            elif provider == "ollama":
                try:
                    import ollama
                    self.client = ollama.Client(host=self.ollama_base_url)
                    # Test connection
                    self.client.list()
                    self.enabled = True
                except Exception:
                    # Fallback to requests-based implementation
                    import requests
                    response = requests.get(f"{self.ollama_base_url}/api/tags")
                    if response.status_code == 200:
                        self.enabled = True
        except ImportError:
            pass
        except Exception:
            pass

    def analyze_column(
        self,
        column_name: str,
        sample_values: List[str],
        dtype: str,
        properties: Dict[str, Any]
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
        except Exception:
            # Fallback to heuristic-based analysis
            return {
                "description": f"Column '{column_name}' of type {dtype}",
                "semantic_type": self._infer_semantic_type(column_name),
                "suggested_validations": [],
                "business_rules": []
            }

    def _create_analysis_prompt(
        self,
        column_name: str,
        sample_values: List[str],
        dtype: str,
        properties: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM analysis."""
        prompt = f"""Analyze this data column and provide insights:

Column Name: {column_name}
Data Type: {dtype}
Sample Values: {', '.join(str(v) for v in sample_values[:10])}
Properties: {json.dumps(properties, indent=2)}

Please provide a JSON response with:
1. "description": A clear, concise description of what this column represents
2. "semantic_type": The semantic meaning (e.g., "customer_id", "email", "price", "timestamp")
3. "suggested_validations": List of additional validation rules that should be applied
4. "business_rules": List of potential business rules or constraints

Respond with valid JSON only."""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM provider."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst expert. Provide JSON responses only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                system="You are a data analyst expert. Provide JSON responses only.",
                temperature=0.1,
                max_tokens=500
            )
            return response.content[0].text

        elif self.provider == "ollama":
            if hasattr(self.client, 'chat'):
                # Using ollama Python package
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst expert. Provide JSON responses only."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.1}
                )
                return response['message']['content']
            else:
                # Using requests fallback
                import requests
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a data analyst expert. Provide JSON responses only."},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {"temperature": 0.1}
                    }
                )
                return response.json()['message']['content']

        return "{}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except Exception:
            # Return empty dict if parsing fails
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

    def generate_documentation(
        self,
        model_name: str,
        columns: List[Dict[str, Any]]
    ) -> str:
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

            if col.get('nullable'):
                doc += "- **Nullable**: Yes\n"
            if col.get('unique'):
                doc += "- **Unique**: Yes\n"
            if col.get('pattern_name'):
                doc += f"- **Pattern**: {col['pattern_name']}\n"
            if col.get('min_value') is not None:
                doc += f"- **Min Value**: {col['min_value']}\n"
            if col.get('max_value') is not None:
                doc += f"- **Max Value**: {col['max_value']}\n"
            if col.get('examples'):
                doc += f"- **Examples**: {', '.join(str(e) for e in col['examples'][:3])}\n"

            doc += "\n"

        return doc