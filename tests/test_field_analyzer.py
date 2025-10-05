"""Tests for the FieldAnalyzer class"""

from numpy import inf
from pandas import DataFrame, Series, to_datetime

from pandera_forge.field_analyzer import FieldAnalyzer


class TestFieldAnalyzer:
    """Test suite for FieldAnalyzer"""

    def test_analyze_numeric_column(self):
        """Test analysis of numeric column"""
        df = DataFrame({"numeric_col": [1, 2, 3, 4, 5]})

        properties = FieldAnalyzer.analyze_column(df, "numeric_col")

        assert properties["is_unique"] is True
        assert properties["is_nullable"] is False
        assert properties["min_value"] == 1
        assert properties["max_value"] == 5
        assert properties["distinct_count"] == 5
        assert len(properties["examples"]) > 0

    def test_analyze_string_column(self):
        """Test analysis of string column"""
        df = DataFrame({"string_col": ["a", "b", "c", "d", "e"]})

        properties = FieldAnalyzer.analyze_column(df, "string_col")

        assert properties["is_unique"] is True
        assert properties["is_nullable"] is False
        assert properties["min_value"] is None  # Not numeric
        assert properties["max_value"] is None  # Not numeric
        assert properties["distinct_count"] == 5
        assert "a" in properties["examples"]

    def test_analyze_nullable_column(self):
        """Test analysis of column with null values"""
        df = DataFrame({"nullable_col": [1, 2, None, 4, 5]})

        properties = FieldAnalyzer.analyze_column(df, "nullable_col")

        assert properties["is_nullable"] is True
        assert properties["min_value"] == 1
        assert properties["max_value"] == 5

    def test_analyze_non_unique_column(self):
        """Test analysis of column with duplicate values"""
        df = DataFrame({"non_unique_col": [1, 1, 2, 2, 3]})

        properties = FieldAnalyzer.analyze_column(df, "non_unique_col")

        assert properties["is_unique"] is False
        assert properties["distinct_count"] == 3

    def test_analyze_boolean_column(self):
        """Test analysis of boolean column"""
        df = DataFrame({"bool_col": [True, False, True, False, True]})

        properties = FieldAnalyzer.analyze_column(df, "bool_col")

        assert properties["is_unique"] is False
        assert properties["is_nullable"] is False
        assert properties["distinct_count"] == 2

    def test_analyze_datetime_column(self):
        """Test analysis of datetime column"""
        df = DataFrame({"date_col": to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])})

        properties = FieldAnalyzer.analyze_column(df, "date_col")

        assert properties["is_unique"] is True
        assert properties["is_nullable"] is False
        assert properties["distinct_count"] == 3

    def test_analyze_unhashable_column(self):
        """Test analysis of column with unhashable types (lists)"""
        df = DataFrame({"list_col": [[1, 2], [3, 4], [5, 6]]})

        properties = FieldAnalyzer.analyze_column(df, "list_col")

        assert properties["is_unique"] is False  # Can't determine uniqueness for unhashable
        assert properties["distinct_count"] is None

    def test_get_examples_most_common(self):
        """Test that examples return most common values"""
        series = Series(["a", "a", "a", "b", "b", "c", "d", "e"])

        examples = FieldAnalyzer._get_examples(series, num_samples=3)

        assert len(examples) == 3
        assert "a" in examples  # Most common
        assert "b" in examples  # Second most common

    def test_get_examples_with_nulls(self):
        """Test getting examples from series with null values"""
        series = Series([1, 2, None, 3, None, 4])

        examples = FieldAnalyzer._get_examples(series)

        assert len(examples) > 0
        assert "None" not in examples or "nan" not in examples

    def test_format_field_properties(self):
        """Test formatting of field properties"""
        properties = {"is_unique": True, "is_nullable": True, "min_value": 1, "max_value": 10}

        formatted = FieldAnalyzer.format_field_properties(properties)

        assert "ge=1" in formatted
        assert "le=10" in formatted
        assert "unique=True" in formatted
        assert "nullable=True" in formatted

    def test_format_field_properties_empty(self):
        """Test formatting when no special properties"""
        properties = {
            "is_unique": False,
            "is_nullable": False,
            "min_value": None,
            "max_value": None,
        }

        formatted = FieldAnalyzer.format_field_properties(properties)

        assert formatted == ""

    def test_infinite_values_handling(self):
        """Test handling of infinite values"""
        df = DataFrame({"inf_col": [1, 2, inf, -inf, 5]})

        properties = FieldAnalyzer.analyze_column(df, "inf_col")

        # Should handle infinite values gracefully
        assert properties["min_value"] == -inf
        assert properties["max_value"] == inf

    def test_all_null_column(self):
        """Test analysis of column with all null values"""
        df = DataFrame({"all_null": [None, None, None]})

        properties = FieldAnalyzer.analyze_column(df, "all_null")

        assert properties["is_nullable"] is True
        assert properties["is_unique"] is False
        assert properties["distinct_count"] == 0
