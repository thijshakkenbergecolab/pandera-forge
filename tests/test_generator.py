"""Tests for the ModelGenerator class"""

from pandas import DataFrame, to_datetime, Categorical
from pandera_forge import ModelGenerator


class TestModelGenerator:
    """Test suite for ModelGenerator"""

    def test_basic_generation(self):
        """Test basic model generation with common data types"""
        df = DataFrame(
            {
                "int_col": [1, 2, 3, 4],
                "float_col": [1.0, 2.0, 3.0, 4.0],
                "str_col": ["a", "b", "c", "d"],
                "bool_col": [True, False, True, False],
                "date_col": to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            }
        )

        generator = ModelGenerator()
        model_code = generator.generate(df, "TestModel")

        assert model_code is not None
        assert "class TestModel(DataFrameModel):" in model_code
        assert "int_col: Series[Int64]" in model_code
        assert "float_col: Series[Float64]" in model_code
        assert "str_col: Series[Object]" in model_code
        assert "bool_col: Series[Bool]" in model_code
        assert "date_col: Series[Timestamp]" in model_code

    def test_nullable_fields(self):
        """Test that nullable fields are properly detected"""
        df = DataFrame({"nullable_col": [1, 2, None, 4], "non_nullable_col": [1, 2, 3, 4]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "NullableModel")

        assert "nullable_col: Series[Float64] = Field(ge=1.0, le=4.0, nullable=True)" in model_code
        assert "nullable=True" in model_code.split("nullable_col")[1].split("\n")[0]
        assert "nullable=True" not in model_code.split("non_nullable_col")[1].split("\n")[0]

    def test_unique_fields(self):
        """Test that unique fields are properly detected"""
        df = DataFrame({"unique_col": [1, 2, 3, 4], "non_unique_col": [1, 1, 2, 2]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "UniqueModel")

        assert "unique=True" in model_code.split("unique_col")[1].split("\n")[0]
        assert "unique=True" not in model_code.split("non_unique_col")[1].split("\n")[0]

    def test_numeric_constraints(self):
        """Test that min/max values are properly set for numeric columns"""
        df = DataFrame({"numeric_col": [10, 20, 30, 40]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "NumericModel")

        assert "ge=10" in model_code
        assert "le=40" in model_code

    def test_column_name_sanitization(self):
        """Test that problematic column names are sanitized"""
        df = DataFrame(
            {
                "Column With Spaces": [1, 2, 3],
                "123_starts_with_number": [4, 5, 6],
                "class": [7, 8, 9],  # Reserved keyword
                "!@#$%": [10, 11, 12],  # Special characters
            }
        )

        generator = ModelGenerator()
        model_code = generator.generate(df, "SanitizedModel")

        assert model_code is not None
        assert 'alias="Column With Spaces"' in model_code
        assert 'alias="123_starts_with_number"' in model_code
        assert 'alias="class"' in model_code
        assert 'alias="!@#$%"' in model_code

    def test_categorical_data(self):
        """Test categorical data type handling"""
        df = DataFrame({"category_col": Categorical(["a", "b", "c", "a", "b"])})

        generator = ModelGenerator()
        model_code = generator.generate(df, "CategoricalModel")

        assert "category_col: Series[Category]" in model_code

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = DataFrame()

        generator = ModelGenerator()
        model_code = generator.generate(df, "EmptyModel")

        assert model_code is None

    def test_numeric_column_names(self):
        """Test handling of numeric column names"""
        df = DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2.5: [7, 8, 9]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "NumericColumnsModel")

        assert model_code is not None
        assert "alias=0" in model_code
        assert "alias=1" in model_code
        assert "alias=2.5" in model_code

    def test_validation_enabled(self):
        """Test that validation works when enabled"""
        df = DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "ValidatedModel", validate=True)

        assert model_code is not None
        # The model should validate successfully against the source DataFrame

    def test_validation_disabled(self):
        """Test that validation can be disabled"""
        df = DataFrame({"col1": [1, 2, 3]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "UnvalidatedModel", validate=False)

        assert model_code is not None

    def test_examples_included(self):
        """Test that examples are included in comments when requested"""
        df = DataFrame({"col1": ["example1", "example2", "example3"]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "ExamplesModel", include_examples=True)

        assert "examples:" in model_code
        assert "example1" in model_code

    def test_examples_excluded(self):
        """Test that examples are excluded when not requested"""
        df = DataFrame({"col1": ["example1", "example2", "example3"]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "NoExamplesModel", include_examples=False)

        assert "examples:" not in model_code

    def test_mixed_types_object_column(self):
        """Test handling of mixed type columns (become Object type)"""
        df = DataFrame({"mixed_col": [1, "two", 3.0, None]})

        generator = ModelGenerator()
        model_code = generator.generate(df, "MixedTypesModel")

        assert "mixed_col: Series[Object]" in model_code
        assert "nullable=True" in model_code
