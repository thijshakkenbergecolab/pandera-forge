"""
Column name sanitization utilities
"""

from keyword import iskeyword
from re import sub
from typing import List, Union


class NameSanitizer:
    """Handles sanitization of column names for valid Python identifiers"""

    INVALID_CHARS: List[str] = list("!@#$%^&*()+={}[]|\\:;'\"<>,/?- ~`.")

    @classmethod
    def sanitize_column_name(cls, name: Union[str, int, float]) -> tuple[str, bool]:
        """
        Sanitize a column name to be a valid Python identifier.

        Returns:
            tuple: (sanitized_name, is_valid) where is_valid indicates if original was valid
        """
        is_valid = True

        # Handle non-string column names
        if not isinstance(name, str):
            is_valid = False
            name = str(name)

        # Replace spaces and non-word characters with underscores
        sanitized = sub(r"\s+", "_", name)
        sanitized = sub(r"\W+", "_", sanitized)

        # Check if sanitization was needed
        if sanitized != name:
            is_valid = False

        # Prefix with 'col_' if name starts with digit or is keyword
        if sanitized and (sanitized[0].isdigit() or iskeyword(sanitized)):
            sanitized = "col_" + sanitized
            is_valid = False

        # Ensure name is not empty
        if not sanitized:
            sanitized = "col_unnamed"
            is_valid = False

        return sanitized, is_valid

    @classmethod
    def sanitize_class_name(cls, name: str) -> str:
        """Sanitize a class name to be a valid Python identifier"""
        # Replace each non-word character with underscore
        sanitized = sub(r"\W", "_", name)

        # Prefix with 'Model' if starts with digit
        if sanitized and sanitized[0].isdigit():
            sanitized = "Model" + sanitized

        # Ensure name is not empty
        if not sanitized:
            sanitized = "DataFrameModel"

        return sanitized
