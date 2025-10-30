"""
Pandas-specific field analysis implementation
"""

from typing import Any, Dict, List
from pandas import DataFrame, Series, notna
from pandas.core.dtypes.common import is_numeric_dtype

from ..core.base_field_analyzer import BaseFieldAnalyzer


class PandasFieldAnalyzer(BaseFieldAnalyzer):
    """Analyzes pandas DataFrame columns to extract properties for Field generation"""

    def analyze_column(self, df: DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Analyze a pandas column and return its properties.

        Args:
            df: Pandas DataFrame
            column_name: Name of the column to analyze

        Returns:
            Dict containing column properties
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
        properties["examples"] = self.get_examples(df, column_name)

        return properties

    def get_column_dtype(self, df: DataFrame, column_name: str) -> str:
        """Get the data type of a pandas column."""
        dtype_str = str(df[column_name].dtype)
        # Handle timestamp types
        if "datetime64" in dtype_str:
            dtype_str = "datetime64[ns]"
        return dtype_str

    def get_examples(self, df: DataFrame, column_name: str, num_samples: int = 5) -> List[Any]:
        """Get the most common values as examples, preserving their original types."""
        column = df[column_name]
        try:
            # Get value counts without converting to string to preserve types
            value_counts = column.value_counts()

            # If there are 10 or fewer unique values, get ALL of them for isin constraint
            if len(value_counts) <= 10:
                return value_counts.index.tolist()
            else:
                # Otherwise get the top N most common values
                top_values = value_counts.nlargest(num_samples)
                return top_values.index.tolist()
        except (TypeError, ValueError):
            # If value_counts fails (e.g., unhashable types), try to get a few non-null samples
            try:
                samples = column.dropna().head(num_samples)
                return samples.tolist()
            except:
                return []

    def is_numeric_column(self, df: DataFrame, column_name: str) -> bool:
        """Check if a pandas column contains numeric data."""
        return is_numeric_dtype(df[column_name])

    def get_column_stats(self, df: DataFrame, column_name: str) -> Dict[str, Any]:
        """Get statistical information about a pandas column."""
        column = df[column_name]
        stats = {}

        if self.is_numeric_column(df, column_name):
            stats["min"] = column.min()
            stats["max"] = column.max()
            stats["mean"] = column.mean()
            stats["median"] = column.median()
            stats["std"] = column.std()
            stats["null_count"] = column.isnull().sum()
            stats["non_null_count"] = column.count()

        return stats