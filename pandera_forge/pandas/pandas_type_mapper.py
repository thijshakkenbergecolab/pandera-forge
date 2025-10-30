"""
Pandas-specific type mapping implementation
"""

from typing import Dict, Optional, Type, Any
from pandera.typing.pandas import (
    Bool,
    Category,
    DateTime,
    Float16,
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


class PandasTypeMapper(BaseTypeMapper):
    """Maps pandas dtypes to Pandera types"""

    PANDAS_TO_PANDERA_MAP: Dict[str, Type] = {
        "int64": Int64,
        "int32": Int32,
        "int16": Int16,
        "int8": Int8,
        "float64": Float64,
        "float32": Float32,
        "float16": Float16,
        "string": String,
        "bool": Bool,
        "datetime64[ns]": DateTime,
        "datetime64": DateTime,
        "category": Category,
        "object": Object,
    }

    def get_type_map(self) -> Dict[str, Type]:
        """Get the pandas to Pandera type mapping dictionary."""
        return self.PANDAS_TO_PANDERA_MAP

    def get_pandera_type(self, dtype: Any) -> Optional[Type]:
        """
        Convert pandas dtype to Pandera type.

        Args:
            dtype: Pandas dtype (string or dtype object)

        Returns:
            Pandera type class or None if no mapping exists
        """
        dtype_str = self.normalize_dtype(dtype)
        return self.PANDAS_TO_PANDERA_MAP.get(dtype_str)

    def normalize_dtype(self, dtype: Any) -> str:
        """
        Normalize pandas dtype to standard string representation.

        Args:
            dtype: Pandas dtype to normalize

        Returns:
            Normalized string representation
        """
        dtype_str = str(dtype)

        # Handle timestamp types
        if "datetime64" in dtype_str:
            return "datetime64[ns]"

        return dtype_str