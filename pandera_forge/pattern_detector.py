"""
Pattern detection and enrichment utilities for string columns
"""

from logging import error
from re import escape
from typing import Any, Dict, Optional, Tuple

from pandas import Series
from pydantic import BaseModel


class StringConstraints(BaseModel):
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    pattern_name: Optional[str] = None
    starts_with: Optional[str] = None
    ends_with: Optional[str] = None
    contains: Optional[list[str]] = None


class PatternDetector:
    """Detects patterns in string columns for enhanced validation"""

    # Common regex patterns - ordered from most specific to least specific
    PATTERNS = {
        # Very specific patterns first
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "ipv4": r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "time_24h": r"^([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$",
        "ssn": r"^\d{3}-\d{2}-\d{4}$",
        "credit_card": r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$",
        "phone_us": r"^\+?1?\d{10,14}$",
        "hex_color": r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
        "mac_address": r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$",
        "json": r"^\{.*\}$|^\[.*\]$",
        # More specific patterns before less specific
        "numeric_string": r"^\d+$",
        "postal_code_us": r"^\d{5}(-\d{4})?$",
        "alphanumeric": r"^[a-zA-Z0-9]+$",
        "alpha_only": r"^[a-zA-Z]+$",
        "slug": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    }

    @classmethod
    def detect_pattern(
        cls, series: Series, min_match_ratio: float = 0.9
    ) -> Optional[Tuple[str, str]]:
        """
        Detect if a series matches any known pattern.

        Args:
            series: Pandas series to analyze
            min_match_ratio: Minimum ratio of values that must match (default 0.9)

        Returns:
            Tuple of (pattern_name, regex) if pattern found, None otherwise
        """
        # Convert to string and drop nulls
        str_series = series.dropna().astype(str)

        if len(str_series) == 0:
            return None

        # Test each pattern
        for pattern_name, pattern_regex in cls.PATTERNS.items():
            try:
                matches = str_series.str.match(pattern_regex, case=False)
                match_ratio = matches.sum() / len(str_series)

                if match_ratio >= min_match_ratio:
                    return pattern_name, pattern_regex
            except Exception as e:
                error(f"Error applying pattern {pattern_name}: {e}")
                continue

        return None

    @classmethod
    def infer_string_constraints(cls, series: Series) -> StringConstraints:
        """
        Infer string-specific constraints from a series.

        Returns:
            Dict with possible constraints:
                - min_length: int
                - max_length: int
                - pattern: str (regex)
                - pattern_name: str (human-readable pattern name)
                - starts_with: str
                - ends_with: str
                - contains: List[str]
        """
        constraints: Dict[str, Any] = {}
        str_series = series.dropna().astype(str)

        if len(str_series) == 0:
            return StringConstraints.model_validate(constraints)

        # Length constraints
        lengths = str_series.str.len()
        constraints["min_length"] = int(lengths.min())
        constraints["max_length"] = int(lengths.max())

        # Pattern detection
        pattern_result = cls.detect_pattern(series)
        if pattern_result:
            constraints["pattern_name"] = pattern_result[0]
            constraints["pattern"] = pattern_result[1]

        # Common prefix/suffix detection
        if len(str_series.unique()) > 1:
            # Check for common prefix
            first_chars = str_series.str[:3].value_counts()
            if len(first_chars) == 1:
                common_prefix = first_chars.index[0]
                if all(str_series.str.startswith(str(common_prefix))):
                    constraints["starts_with"] = str(common_prefix)

            # Check for common suffix
            last_chars = str_series.str[-3:].value_counts()
            if len(last_chars) == 1:
                common_suffix = last_chars.index[0]
                if all(str_series.str.endswith(str(common_suffix))):
                    constraints["ends_with"] = str(common_suffix)

        return StringConstraints.model_validate(constraints)

    @classmethod
    def generate_custom_regex(cls, series: Series, sample_size: int = 100) -> Optional[str]:
        """
        Attempt to generate a custom regex pattern from sample values.

        Args:
            series: Pandas series to analyze
            sample_size: Number of samples to use for pattern generation

        Returns:
            Regex pattern string or None
        """
        str_series = series.dropna().astype(str)

        if len(str_series) == 0:
            return None

        # Take a sample for analysis
        sample = str_series.head(sample_size)

        # Simple pattern inference based on character classes
        patterns = []
        for value in sample:
            if not value:
                continue

            char_pattern = ""
            for char in str(value):
                if char.isdigit():
                    char_pattern += r"\d"
                elif char.isalpha():
                    if char.isupper():
                        char_pattern += "[A-Z]"
                    else:
                        char_pattern += "[a-z]"
                elif char in ".-_":
                    char_pattern += "\\" + char
                elif char == " ":
                    char_pattern += r"\s"
                else:
                    char_pattern += escape(char)

            patterns.append(char_pattern)

        # Find the most common pattern
        if patterns:
            from collections import Counter

            pattern_counts = Counter(patterns)
            most_common = pattern_counts.most_common(1)[0]

            # Only return if pattern is consistent enough
            if most_common[1] / len(patterns) >= 0.8:
                return "^" + most_common[0] + "$"

        return None
