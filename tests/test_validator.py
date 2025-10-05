"""Tests for validator module"""

from pandas import DataFrame

from pandera_forge.validator import ModelValidator


class TestModelValidator:
    """Test suite for ModelValidator class"""

    def test_validate_model_code_success(self):
        """Test successful validation of syntactically correct model code"""
        model_code = """
class TestModel(DataFrameModel):
    col1: Series[Int64] = Field()
    col2: Series[String] = Field()
"""

        is_valid, error = ModelValidator.validate_model_code(model_code, "TestModel")

        assert is_valid is True
        assert error is None

    def test_validate_model_code_syntax_error(self):
        """Test validation with syntax error"""
        model_code = """
class TestModel(DataFrameModel):
    col1: Series[Int64] = Field(
    col2: Series[String] = Field()
"""

        is_valid, error = ModelValidator.validate_model_code(model_code, "TestModel")

        assert is_valid is False
        assert "Syntax error" in error

    def test_validate_model_code_class_not_found(self):
        """Test validation when class is not found in code"""
        model_code = """
class WrongModel(DataFrameModel):
    col1: Series[Int64] = Field()
"""

        is_valid, error = ModelValidator.validate_model_code(model_code, "TestModel")

        assert is_valid is False
        assert "Class TestModel not found" in error

    def test_validate_model_code_execution_error(self):
        """Test validation with execution error"""
        model_code = """
class TestModel(DataFrameModel):
    col1: Series[Int64] = Field()
    col2: Series[String] = Field(invalid_param=True)
"""

        is_valid, error = ModelValidator.validate_model_code(model_code, "TestModel")

        assert is_valid is False
        assert "Execution error" in error

    def test_validate_against_dataframe_success(self):
        """Test successful validation against DataFrame"""
        df = DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        model_code = """
class TestModel(DataFrameModel):
    col1: Series[Int64] = Field()
    col2: Series[Object] = Field()
"""

        is_valid, error = ModelValidator.validate_against_dataframe(model_code, "TestModel", df)

        assert is_valid is True
        assert error is None

    def test_validate_against_dataframe_class_not_found(self):
        """Test validation when class not found"""
        df = DataFrame({"col1": [1, 2, 3]})

        model_code = """
class WrongModel(DataFrameModel):
    col1: Series[Int64] = Field()
"""

        is_valid, error = ModelValidator.validate_against_dataframe(model_code, "TestModel", df)

        assert is_valid is False
        assert "Class TestModel not found" in error

    def test_validate_against_dataframe_validation_error(self):
        """Test validation failure against DataFrame"""
        df = DataFrame({"col1": ["not", "an", "integer"]})  # String values for Int64 field

        model_code = """
class TestModel(DataFrameModel):
    col1: Series[Int64] = Field()
"""

        is_valid, error = ModelValidator.validate_against_dataframe(model_code, "TestModel", df)

        assert is_valid is False
        assert "Validation error" in error
