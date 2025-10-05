"""
Field analysis utilities for extracting column properties
"""

from typing import Any, Dict, List
from pandas import DataFrame, Series, notna
from pandas.core.dtypes.common import is_numeric_dtype


class FieldAnalyzer:
    """Analyzes DataFrame columns to extract properties for Field generation"""

    @staticmethod
    def analyze_column(df: DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Analyze a column and return its properties.

        Returns:
            Dict containing:
                - is_unique: bool
                - is_nullable: bool
                - min_value: numeric or None
                - max_value: numeric or None
                - examples: List[Any]
                - distinct_count: int or None
        """
        column = df[column_name]
        properties: Dict[str, Any] = {}

        # Check uniqueness
        try:
            # For uniqueness check, we need to consider nulls
            # If there are nulls and non-null values are unique, still not fully unique
            non_null_count = column.count()
            # Convert to bool to avoid numpy.bool_ type
            properties["is_unique"] = bool(
                column.nunique() == len(df) and non_null_count == len(df)
            )
            properties["distinct_count"] = int(column.nunique())
        except TypeError:  # handles unhashable types
            properties["is_unique"] = False
            properties["distinct_count"] = None

        # Check nullability
        properties["is_nullable"] = bool(column.isnull().any())

        # Get min/max for numeric types
        if is_numeric_dtype(column):
            min_val = column.min()
            max_val = column.max()
            # Only set if not NaN
            properties["min_value"] = min_val if notna(min_val) else None
            properties["max_value"] = max_val if notna(max_val) else None
        else:
            properties["min_value"] = None
            properties["max_value"] = None

        # Get examples
        properties["examples"] = FieldAnalyzer._get_examples(column)

        return properties

    @staticmethod
    def _get_examples(column: Series, num_samples: int = 5) -> List[Any]:
        """Get the most common values as examples, preserving their original types"""
        try:
            # Get value counts without converting to string to preserve types
            value_counts = column.value_counts()
            top_values = value_counts.nlargest(num_samples)
            return top_values.index.tolist()
        except (TypeError, ValueError):
            # If value_counts fails (e.g., unhashable types), try to get a few non-null samples
            try:
                samples = column.dropna().head(num_samples)
                return samples.tolist()
            except:
                return []

    @staticmethod
    def format_field_properties(properties: Dict[str, Any]) -> str:
        """Format properties dict into Field() parameter string"""
        field_params = []

        if properties.get("is_unique"):
            field_params.append("unique=True")

        if properties.get("is_nullable"):
            field_params.append("nullable=True")

        # Add min/max for numeric types
        if properties.get("min_value") is not None and properties.get("max_value") is not None:
            # Check if values are finite
            min_val = properties["min_value"]
            max_val = properties["max_value"]
            if notna(min_val) and notna(max_val):
                field_params.insert(0, f"ge={min_val}")
                field_params.insert(1, f"le={max_val}")

        return ", ".join(field_params) if field_params else ""
