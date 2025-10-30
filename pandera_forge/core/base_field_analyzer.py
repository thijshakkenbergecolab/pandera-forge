"""
Abstract base class for field analysis across different DataFrame implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseFieldAnalyzer(ABC):
    """Abstract base class for analyzing DataFrame columns to extract properties"""

    @abstractmethod
    def analyze_column(self, df: Any, column_name: str) -> Dict[str, Any]:
        """
        Analyze a column and return its properties.

        Args:
            df: The DataFrame (pandas, Spark, etc.)
            column_name: Name of the column to analyze

        Returns:
            Dict containing:
                - is_unique: bool
                - is_nullable: bool
                - min_value: numeric or None
                - max_value: numeric or None
                - examples: List[Any]
                - distinct_count: int or None
                - Additional properties specific to implementation
        """
        pass

    @abstractmethod
    def get_column_dtype(self, df: Any, column_name: str) -> str:
        """
        Get the data type of a column.

        Args:
            df: The DataFrame
            column_name: Name of the column

        Returns:
            String representation of the column's data type
        """
        pass

    @abstractmethod
    def get_examples(self, df: Any, column_name: str, num_samples: int = 5) -> List[Any]:
        """
        Get example values from a column.

        Args:
            df: The DataFrame
            column_name: Name of the column
            num_samples: Number of samples to retrieve

        Returns:
            List of example values from the column
        """
        pass

    @abstractmethod
    def is_numeric_column(self, df: Any, column_name: str) -> bool:
        """
        Check if a column contains numeric data.

        Args:
            df: The DataFrame
            column_name: Name of the column

        Returns:
            True if the column is numeric, False otherwise
        """
        pass

    @abstractmethod
    def get_column_stats(self, df: Any, column_name: str) -> Dict[str, Any]:
        """
        Get statistical information about a column.

        Args:
            df: The DataFrame
            column_name: Name of the column

        Returns:
            Dictionary with statistical information (min, max, mean, etc.)
        """
        pass

    def format_field_properties(self, properties: Dict[str, Any]) -> str:
        """
        Format properties dict into Field() parameter string.
        This is common across implementations.

        Args:
            properties: Dictionary of field properties

        Returns:
            Formatted string for Field() parameters
        """
        field_params = []

        if properties.get("is_unique"):
            field_params.append("unique=True")

        if properties.get("is_nullable"):
            field_params.append("nullable=True")

        # Add min/max for numeric types
        if properties.get("min_value") is not None and properties.get("max_value") is not None:
            field_params.insert(0, f"ge={properties['min_value']}")
            field_params.insert(1, f"le={properties['max_value']}")

        return ", ".join(field_params) if field_params else ""