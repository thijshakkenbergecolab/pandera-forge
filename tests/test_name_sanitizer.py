"""Tests for the NameSanitizer class"""

import pytest
from pandera_forge.name_sanitizer import NameSanitizer


class TestNameSanitizer:
    """Test suite for NameSanitizer"""

    def test_valid_column_name(self):
        """Test that valid column names are not modified"""
        name, is_valid = NameSanitizer.sanitize_column_name("valid_column_name")
        assert name == "valid_column_name"
        assert is_valid is True

    def test_column_name_with_spaces(self):
        """Test sanitization of column names with spaces"""
        name, is_valid = NameSanitizer.sanitize_column_name("Column With Spaces")
        assert name == "Column_With_Spaces"
        assert is_valid is False

    def test_column_name_with_special_chars(self):
        """Test sanitization of column names with special characters"""
        name, is_valid = NameSanitizer.sanitize_column_name("!@#$%^&*()")
        assert "_" in name
        assert is_valid is False
        # Should not contain any special characters
        assert all(c.isalnum() or c == "_" for c in name)

    def test_column_name_starting_with_digit(self):
        """Test sanitization of column names starting with digit"""
        name, is_valid = NameSanitizer.sanitize_column_name("123column")
        assert name == "col_123column"
        assert is_valid is False

    def test_reserved_keyword_column_name(self):
        """Test sanitization of reserved Python keywords"""
        name, is_valid = NameSanitizer.sanitize_column_name("class")
        assert name == "col_class"
        assert is_valid is False

        name, is_valid = NameSanitizer.sanitize_column_name("def")
        assert name == "col_def"
        assert is_valid is False

    def test_numeric_column_name(self):
        """Test handling of numeric column names"""
        name, is_valid = NameSanitizer.sanitize_column_name(123)
        assert name == "col_123"
        assert is_valid is False

        name, is_valid = NameSanitizer.sanitize_column_name(45.67)
        assert name == "col_45_67"
        assert is_valid is False

    def test_empty_string_column_name(self):
        """Test handling of empty string column name"""
        name, is_valid = NameSanitizer.sanitize_column_name("")
        assert name == "col_unnamed"
        assert is_valid is False

    def test_class_name_sanitization(self):
        """Test sanitization of class names"""
        name = NameSanitizer.sanitize_class_name("Valid Class Name")
        assert name == "Valid_Class_Name"

    def test_class_name_with_special_chars(self):
        """Test class name with special characters"""
        name = NameSanitizer.sanitize_class_name("My-Model!@#")
        assert name == "My_Model___"
        assert all(c.isalnum() or c == "_" for c in name)

    def test_class_name_starting_with_digit(self):
        """Test class name starting with digit"""
        name = NameSanitizer.sanitize_class_name("123Model")
        assert name == "Model123Model"
        assert not name[0].isdigit()

    def test_empty_class_name(self):
        """Test handling of empty class name"""
        name = NameSanitizer.sanitize_class_name("")
        assert name == "DataFrameModel"

    def test_mixed_case_preservation(self):
        """Test that mixed case is preserved when possible"""
        name, is_valid = NameSanitizer.sanitize_column_name("CamelCaseColumn")
        assert name == "CamelCaseColumn"
        assert is_valid is True

    def test_underscore_preservation(self):
        """Test that underscores are preserved"""
        name, is_valid = NameSanitizer.sanitize_column_name("column_with_underscores")
        assert name == "column_with_underscores"
        assert is_valid is True

    def test_hyphen_replacement(self):
        """Test that hyphens are replaced with underscores"""
        name, is_valid = NameSanitizer.sanitize_column_name("column-with-hyphens")
        assert name == "column_with_hyphens"
        assert is_valid is False

    def test_dot_replacement(self):
        """Test that dots are replaced with underscores"""
        name, is_valid = NameSanitizer.sanitize_column_name("column.with.dots")
        assert name == "column_with_dots"
        assert is_valid is False