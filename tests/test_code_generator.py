"""Tests for code_generator module"""

from numpy import nan

from pandera_forge.code_generator import CodeGenerator


class TestCodeGenerator:
    """Test suite for CodeGenerator class"""

    def test_generate_comment_with_examples_and_distinct(self):
        """Test comment generation with examples and distinct count"""
        properties = {"examples": ["val1", "val2", "val3"], "distinct_count": 10}

        comment = CodeGenerator.generate_comment(properties)
        assert "10 distinct values" in comment
        assert "examples:" in comment
        assert "val1" in comment

    def test_generate_comment_with_examples_only(self):
        """Test comment generation with only examples"""
        properties = {"examples": [1, 2, 3]}

        comment = CodeGenerator.generate_comment(properties)
        assert "examples:" in comment
        assert "1" in comment
        assert "distinct values" not in comment

    def test_generate_comment_empty_properties(self):
        """Test comment generation with empty properties"""
        properties = {}

        comment = CodeGenerator.generate_comment(properties)
        assert comment == ""

    def test_generate_field_string_with_nan_in_examples(self):
        """Test field generation with NaN values in examples"""
        properties = {"examples": [1.0, nan, 3.0], "is_nullable": True, "distinct_count": 3}

        field_str = CodeGenerator.generate_field_string(
            field_name="test_field",
            pandera_type="Float64",
            properties=properties,
            original_column_name="test_field",
            needs_alias=False,
        )

        # NaN should be skipped in isin list
        assert "isin=[1.0, 3.0]" in field_str  # noqa isin is the name of the parameter
        assert "nan" not in field_str.lower()

    def test_generate_field_string_with_alias_numeric(self):
        """Test field generation with numeric column name needing alias"""
        properties = {}

        field_str = CodeGenerator.generate_field_string(
            field_name="col_0",
            pandera_type="Int64",
            properties=properties,
            original_column_name=0,
            needs_alias=True,
        )

        assert "alias=0" in field_str
        assert 'alias="0"' not in field_str

    def test_generate_field_string_with_alias_string(self):
        """Test field generation with string column name needing alias"""
        properties = {}

        field_str = CodeGenerator.generate_field_string(
            field_name="my_column",
            pandera_type="String",
            properties=properties,
            original_column_name="my-column",
            needs_alias=True,
        )

        assert 'alias="my-column"' in field_str

    def test_generate_class_definition_empty_fields(self):
        """Test class definition with no fields"""
        class_def = CodeGenerator.generate_class_definition("EmptyModel", [])

        assert "class EmptyModel(DataFrameModel):" in class_def
        assert "\tpass" in class_def

    def test_generate_class_definition_with_fields(self):
        """Test class definition with fields"""
        fields = ["\tfield1: Series[Int64] = Field()", "\tfield2: Series[String] = Field()"]

        class_def = CodeGenerator.generate_class_definition("TestModel", fields)

        assert "class TestModel(DataFrameModel):" in class_def
        assert "field1: Series[Int64]" in class_def
        assert "field2: Series[String]" in class_def
        assert "pass" not in class_def

    def test_generate_imports(self):
        """Test import generation uses the constant"""
        imports = CodeGenerator.generate_imports()

        assert "from pandera.pandas import" in imports
        assert "from pandera.typing.pandas import" in imports
        assert "from typing import Optional" in imports
