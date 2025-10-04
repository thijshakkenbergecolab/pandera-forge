"""
Code generation utilities for creating Pandera model strings
"""

from typing import Any, Dict, Optional, Union

from pandas import notna, isna


class CodeGenerator:
    """Generates Python code strings for Pandera models"""

    @staticmethod
    def generate_field_string(
        field_name: str,
        pandera_type: str,
        properties: Dict[str, Any],
        original_column_name: Union[str, int],
        needs_alias: bool,
    ) -> str:
        """
        Generate a Field definition string.

        Args:
            field_name: Sanitized field name
            pandera_type: Pandera type string (e.g., "Int64")
            properties: Field properties dict from FieldAnalyzer
            original_column_name: Original column name for alias
            needs_alias: Whether to add alias parameter

        Returns:
            Complete field definition string
        """
        # Format field parameters
        field_params = []

        # Add numeric constraints if present
        if properties.get("min_value") is not None and properties.get("max_value") is not None:
            min_val = properties["min_value"]
            max_val = properties["max_value"]
            if notna(min_val) and notna(max_val):
                field_params.append(f"ge={min_val}")
                field_params.append(f"le={max_val}")

        # Add other properties
        if properties.get("is_unique"):
            field_params.append("unique=True")

        if properties.get("is_nullable"):
            field_params.append("nullable=True")

        # Add examples as isin parameter if present
        examples = properties.get("examples", [])
        if examples:
            # Format examples based on their type
            formatted_examples = []
            for ex in examples:
                if ex is None or (isinstance(ex, float) and isna(ex)):
                    # Skip None/NaN values in isin list
                    continue
                elif isinstance(ex, str):
                    formatted_examples.append(f'"{ex}"')
                elif isinstance(ex, (int, float, bool)):
                    # Keep numeric and boolean values unquoted
                    formatted_examples.append(str(ex))
                else:
                    # For other types, convert to string and quote
                    formatted_examples.append(f"{str(ex)}")

            if formatted_examples:
                field_params.append(f"isin=[{', '.join(formatted_examples)}]")

        # Build field string
        params_str = ", ".join(field_params) if field_params else ""

        # Add alias if needed
        if needs_alias:
            if params_str:
                params_str += ", "
            if isinstance(original_column_name, str):
                params_str += f'alias="{original_column_name}"'
            else:
                params_str += f"alias={original_column_name}"

        field_str = f"\t{field_name}: Series[{pandera_type}] = Field({params_str})"

        return field_str

    @staticmethod
    def generate_comment(properties: Dict[str, Any]) -> str:
        """Generate comment with examples and statistics"""
        examples = properties.get("examples", [])
        distinct_count = properties.get("distinct_count")

        if examples and distinct_count is not None:
            examples_str = ", ".join(
                [f'"{ex}"' if isinstance(ex, str) else str(ex) for ex in examples[:5]]
            )
            return f"  # {distinct_count} distinct values, examples: [{examples_str}]"
        elif examples:
            examples_str = ", ".join(
                [f'"{ex}"' if isinstance(ex, str) else str(ex) for ex in examples[:5]]
            )
            return f"  # examples: [{examples_str}]"
        return ""

    @staticmethod
    def generate_imports() -> str:
        """Generate import statements for the model"""
        return """from pandera.pandas import Timestamp, DataFrameModel, Field
from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional"""

    @staticmethod
    def generate_class_definition(class_name: str, fields: list[str]) -> str:
        """Generate complete class definition"""
        class_str = f"class {class_name}(DataFrameModel):\n"
        class_str += "\n".join(fields)
        if not fields:
            class_str += "\tpass"
        return class_str
