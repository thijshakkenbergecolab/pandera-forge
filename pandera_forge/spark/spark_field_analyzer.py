"""
Spark-specific field analysis implementation
"""

from typing import Any, Dict, List, Optional

from ..core.base_field_analyzer import BaseFieldAnalyzer


class SparkFieldAnalyzer(BaseFieldAnalyzer):
    """Analyzes Spark DataFrame columns to extract properties for Field generation"""

    def __init__(self, sample_size: Optional[int] = 10000):
        """
        Initialize the Spark field analyzer.

        Args:
            sample_size: Number of rows to sample for analysis (None for full DataFrame)
        """
        self.sample_size = sample_size

    def analyze_column(self, df: Any, column_name: str) -> Dict[str, Any]:
        """
        Analyze a Spark column and return its properties.

        Args:
            df: Spark DataFrame
            column_name: Name of the column to analyze

        Returns:
            Dict containing column properties
        """
        properties: Dict[str, Any] = {}

        try:
            from pyspark.sql import functions as F

            # Check nullability
            null_count = df.filter(F.col(column_name).isNull()).count()
            properties["is_nullable"] = null_count > 0

            # Check uniqueness (expensive operation, use sampling)
            total_count = df.count()
            distinct_count = df.select(column_name).distinct().count()
            properties["is_unique"] = (distinct_count == total_count) and (null_count == 0)
            properties["distinct_count"] = distinct_count

            # Get column stats for numeric types
            if self.is_numeric_column(df, column_name):
                stats = df.select(
                    F.min(column_name).alias("min"),
                    F.max(column_name).alias("max")
                ).collect()[0]

                properties["min_value"] = stats["min"]
                properties["max_value"] = stats["max"]
            else:
                properties["min_value"] = None
                properties["max_value"] = None

            # Get examples
            properties["examples"] = self.get_examples(df, column_name)

        except ImportError:
            # PySpark not installed, return minimal properties
            properties = {
                "is_nullable": True,
                "is_unique": False,
                "distinct_count": None,
                "min_value": None,
                "max_value": None,
                "examples": []
            }

        return properties

    def get_column_dtype(self, df: Any, column_name: str) -> str:
        """Get the data type of a Spark column."""
        try:
            # Get the schema field for this column
            field = [f for f in df.schema.fields if f.name == column_name][0]
            # Return the simple string representation
            if hasattr(field.dataType, 'simpleString'):
                return field.dataType.simpleString()
            else:
                return str(field.dataType)
        except:
            return "unknown"

    def get_examples(self, df: Any, column_name: str, num_samples: int = 5) -> List[Any]:
        """
        Get example values from a Spark column.

        Args:
            df: Spark DataFrame
            column_name: Name of the column
            num_samples: Number of samples to retrieve

        Returns:
            List of example values
        """
        try:
            from pyspark.sql import functions as F

            # For small distinct counts, get all values
            distinct_df = df.select(column_name).distinct()
            distinct_count = distinct_df.count()

            if distinct_count <= 10:
                # Get all distinct values
                rows = distinct_df.filter(F.col(column_name).isNotNull()).collect()
                return [row[0] for row in rows]
            else:
                # Get most common values using groupBy
                value_counts = (
                    df.groupBy(column_name)
                    .count()
                    .filter(F.col(column_name).isNotNull())
                    .orderBy(F.desc("count"))
                    .limit(num_samples)
                    .collect()
                )
                return [row[column_name] for row in value_counts]

        except:
            # Fallback: just get a few samples
            try:
                rows = df.select(column_name).filter(
                    df[column_name].isNotNull()
                ).limit(num_samples).collect()
                return [row[0] for row in rows]
            except:
                return []

    def is_numeric_column(self, df: Any, column_name: str) -> bool:
        """Check if a Spark column contains numeric data."""
        dtype = self.get_column_dtype(df, column_name).lower()
        numeric_types = [
            "byte", "short", "int", "integer", "long", "bigint",
            "float", "double", "decimal"
        ]
        return any(t in dtype for t in numeric_types)

    def get_column_stats(self, df: Any, column_name: str) -> Dict[str, Any]:
        """Get statistical information about a Spark column."""
        stats = {}

        try:
            from pyspark.sql import functions as F

            if self.is_numeric_column(df, column_name):
                # Use Spark's describe() for numeric columns
                stats_df = df.select(column_name).describe()
                stats_dict = {row['summary']: row[column_name] for row in stats_df.collect()}

                stats["count"] = stats_dict.get("count")
                stats["mean"] = stats_dict.get("mean")
                stats["stddev"] = stats_dict.get("stddev")
                stats["min"] = stats_dict.get("min")
                stats["max"] = stats_dict.get("max")

                # Get null count separately
                stats["null_count"] = df.filter(F.col(column_name).isNull()).count()
                stats["non_null_count"] = df.filter(F.col(column_name).isNotNull()).count()

        except:
            pass

        return stats

    def sample_dataframe(self, df: Any) -> Any:
        """
        Sample the DataFrame for faster analysis if needed.

        Args:
            df: Spark DataFrame

        Returns:
            Sampled DataFrame or original if sampling not needed
        """
        if self.sample_size is None:
            return df

        try:
            total_count = df.count()
            if total_count > self.sample_size:
                fraction = self.sample_size / total_count
                return df.sample(withReplacement=False, fraction=fraction)
        except:
            pass

        return df