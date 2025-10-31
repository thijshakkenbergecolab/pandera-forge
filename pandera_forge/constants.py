"""Constants used throughout the Pandera Forge package."""

# Common Pandera imports used in generated models and validation
# Try modern imports first, fall back to legacy for compatibility
PANDERA_IMPORTS = """
try:
    from pandera import DataFrameModel, Field
    from pandera.typing import Timestamp
    from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
except ImportError:
    # Fallback for older Pandera versions
    from pandera.pandas import DataFrameModel, Field, Timestamp
    from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional
"""
