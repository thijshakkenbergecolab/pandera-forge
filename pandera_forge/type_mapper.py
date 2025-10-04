"""
Type mapping utilities for converting pandas dtypes to pandera types
"""

from typing import Dict, Optional, Type
from pandera.typing import (
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


class TypeMapper:
    """Maps pandas dtypes to pandera types"""

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

    @classmethod
    def get_pandera_type(cls, dtype: str) -> Optional[Type]:
        """Convert pandas dtype string to pandera type"""
        return cls.PANDAS_TO_PANDERA_MAP.get(dtype)