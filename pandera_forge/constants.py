"""Constants used throughout the Pandera Forge package."""

# Common Pandera imports used in generated models and validation
PANDERA_IMPORTS = """from pandera.pandas import Timestamp, DataFrameModel, Field
from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional
"""
