"""
Spark-specific type mapping implementation
"""

from typing import Dict, Optional, Type, Any
from pandera.typing.pandas import (
    Bool,
    DateTime as Timestamp,  # DateTime is actually Timestamp in pandera
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Object,
    String,
)

from ..core.base_type_mapper import BaseTypeMapper


class SparkTypeMapper(BaseTypeMapper):
    """Maps Spark SQL types to Pandera types"""

    # Mapping from Spark SQL type names to Pandera types
    SPARK_TO_PANDERA_MAP: Dict[str, Type] = {
        # Numeric types
        "byte": Int8,
        "tinyint": Int8,
        "short": Int16,
        "smallint": Int16,
        "int": Int32,
        "integer": Int32,
        "long": Int64,
        "bigint": Int64,
        "float": Float32,
        "real": Float32,
        "double": Float64,
        "decimal": Float64,  # Approximate mapping
        # String types
        "string": String,
        "varchar": String,
        "char": String,
        # Boolean
        "boolean": Bool,
        "bool": Bool,
        # Date/Time types
        "date": Timestamp,
        "timestamp": Timestamp,
        "timestamp_ntz": Timestamp,
        "timestamp_ltz": Timestamp,
        # Binary and complex types
        "binary": Object,
        "array": Object,
        "map": Object,
        "struct": Object,
    }

    def get_type_map(self) -> Dict[str, Type]:
        """Get the Spark to Pandera type mapping dictionary."""
        return self.SPARK_TO_PANDERA_MAP

    def get_pandera_type(self, dtype: Any) -> Optional[Type]:
        """
        Convert Spark SQL type to Pandera type.

        Args:
            dtype: Spark SQL type (string, DataType object, or StructField)

        Returns:
            Pandera type class or None if no mapping exists
        """
        dtype_str = self.normalize_dtype(dtype)
        return self.SPARK_TO_PANDERA_MAP.get(dtype_str)

    def normalize_dtype(self, dtype: Any) -> str:
        """
        Normalize Spark dtype to standard string representation.

        Args:
            dtype: Spark dtype to normalize (can be string, DataType, or StructField)

        Returns:
            Normalized string representation
        """
        # If it's already a string, clean it up
        if isinstance(dtype, str):
            dtype_str = dtype.lower()
        else:
            # Try to handle pyspark.sql.types objects
            try:
                # For DataType objects, get simpleString()
                if hasattr(dtype, 'simpleString'):
                    dtype_str = dtype.simpleString().lower()
                # For StructField objects, get the dataType
                elif hasattr(dtype, 'dataType'):
                    if hasattr(dtype.dataType, 'simpleString'):
                        dtype_str = dtype.dataType.simpleString().lower()
                    else:
                        dtype_str = str(dtype.dataType).lower()
                else:
                    dtype_str = str(dtype).lower()
            except:
                dtype_str = str(dtype).lower()

        # Handle parameterized types (e.g., "decimal(10,2)" -> "decimal")
        if "(" in dtype_str:
            dtype_str = dtype_str.split("(")[0]

        # Handle array types (e.g., "array<string>" -> "array")
        if "<" in dtype_str:
            dtype_str = dtype_str.split("<")[0]

        return dtype_str

    def get_spark_type_from_string(self, type_string: str):
        """
        Convert a string representation to a Spark SQL type.
        This is useful when creating schemas programmatically.

        Args:
            type_string: String representation of the type

        Returns:
            pyspark.sql.types DataType object or None
        """
        try:
            from pyspark.sql import types as T

            type_map = {
                "string": T.StringType(),
                "int": T.IntegerType(),
                "integer": T.IntegerType(),
                "long": T.LongType(),
                "bigint": T.LongType(),
                "float": T.FloatType(),
                "double": T.DoubleType(),
                "boolean": T.BooleanType(),
                "bool": T.BooleanType(),
                "date": T.DateType(),
                "timestamp": T.TimestampType(),
                "binary": T.BinaryType(),
                "byte": T.ByteType(),
                "short": T.ShortType(),
            }
            return type_map.get(type_string.lower())
        except ImportError:
            # PySpark not installed
            return None