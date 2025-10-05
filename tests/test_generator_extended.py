"""Extended tests for ModelGenerator to improve coverage"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pytest
from pandas import DataFrame, Series

from pandera_forge.generator import ModelGenerator
from pandera_forge.llm_enricher import LLMEnricher


class TestModelGeneratorExtended:
    """Extended tests for ModelGenerator class"""

    def test_generator_with_llm_enricher_instance(self):
        """Test generator initialization with pre-configured LLMEnricher"""
        mock_enricher = Mock(spec=LLMEnricher)
        generator = ModelGenerator(llm_enricher=mock_enricher)
        assert generator.llm_enricher == mock_enricher

    def test_generator_with_llm_api_key(self):
        """Test generator initialization with API key"""
        with patch("pandera_forge.generator.LLMEnricher") as mock_enricher_class:
            generator = ModelGenerator(llm_api_key="test-key")
            mock_enricher_class.assert_called_once_with(api_key="test-key")

    def test_generate_validation_syntax_error(self):
        """Test generate when validation finds syntax errors"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        generator = ModelGenerator()

        with patch.object(generator.validator, "validate_model_code") as mock_validate:
            mock_validate.return_value = (False, "Syntax error")

            result = generator.generate(df, "TestModel", validate=True)
            assert result is None

    def test_generate_validation_dataframe_error(self):
        """Test generate when DataFrame validation fails"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        generator = ModelGenerator()

        with patch.object(generator.validator, "validate_model_code") as mock_syntax:
            mock_syntax.return_value = (True, None)

            with patch.object(generator.validator, "validate_against_dataframe") as mock_df_validate:
                mock_df_validate.return_value = (False, "Validation failed")

                result = generator.generate(df, "TestModel", validate=True)
                assert result is None

    def test_generate_with_source_file(self):
        """Test generate with source_file parameter"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        generator = ModelGenerator()
        source_file = Path("/test/data.csv")

        result = generator.generate(df, "TestModel", validate=False, source_file=source_file)

        assert result is not None
        assert "# Example implementation" in result
        assert "data.csv" in result

    def test_from_csv_success(self):
        """Test from_csv with valid CSV file"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n")
            f.write("1,a\n")
            f.write("2,b\n")
            csv_path = Path(f.name)

        try:
            generator = ModelGenerator()
            result = generator.from_csv(csv_path, validate=False)

            assert result is not None
            assert "col1" in result
            assert "col2" in result
        finally:
            csv_path.unlink()

    def test_from_csv_unicode_error(self):
        """Test from_csv with Unicode error and fallback to latin-1"""
        csv_path = Path("/test/data.csv")
        generator = ModelGenerator()

        with patch("pandera_forge.generator.read_csv") as mock_read:
            # First call raises UnicodeError, second succeeds with latin-1
            mock_read.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "test"),
                pd.DataFrame({"col1": [1, 2, 3]})
            ]

            with patch.object(generator, "generate") as mock_generate:
                mock_generate.return_value = "model code"

                result = generator.from_csv(csv_path, validate=False)

                assert result == "model code"
                assert mock_read.call_count == 2
                # Check that second call used latin-1 encoding
                mock_read.assert_called_with(csv_path, encoding="latin-1")

    def test_from_csv_latin1_also_fails(self):
        """Test from_csv when both UTF-8 and latin-1 fail"""
        csv_path = Path("/test/data.csv")
        generator = ModelGenerator()

        with patch("pandera_forge.generator.read_csv") as mock_read:
            mock_read.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "test"),
                Exception("Latin-1 also failed")
            ]

            result = generator.from_csv(csv_path, validate=False)
            assert result is None

    def test_from_csv_general_error(self):
        """Test from_csv with general read error"""
        csv_path = Path("/test/data.csv")
        generator = ModelGenerator()

        with patch("pandera_forge.generator.read_csv") as mock_read:
            mock_read.side_effect = Exception("File not found")

            result = generator.from_csv(csv_path, validate=False)
            assert result is None

    def test_from_excel_single_sheet(self):
        """Test from_excel with single sheet Excel file"""
        xlsx_path = Path("/test/data.xlsx")
        generator = ModelGenerator()

        mock_excel = MagicMock()
        mock_excel.sheet_names = ["Sheet1"]
        mock_excel.parse.return_value = pd.DataFrame({"col1": [1, 2, 3]})

        with patch("pandera_forge.generator.ExcelFile") as mock_excel_class:
            mock_excel_class.return_value = mock_excel

            with patch.object(generator, "generate") as mock_generate:
                mock_generate.return_value = "model code"

                result = generator.from_excel(xlsx_path, validate=False)

                assert "data" in result
                assert result["data"] == "model code"

    def test_from_excel_multiple_sheets(self):
        """Test from_excel with multiple sheets"""
        xlsx_path = Path("/test/data.xlsx")
        generator = ModelGenerator()

        mock_excel = MagicMock()
        mock_excel.sheet_names = ["Sheet1", "Sheet2"]
        mock_excel.parse.side_effect = [
            pd.DataFrame({"col1": [1, 2, 3]}),
            pd.DataFrame({"col2": ["a", "b", "c"]})
        ]

        with patch("pandera_forge.generator.ExcelFile") as mock_excel_class:
            mock_excel_class.return_value = mock_excel

            with patch.object(generator, "generate") as mock_generate:
                mock_generate.side_effect = ["model1", "model2"]

                result = generator.from_excel(xlsx_path, validate=False)

                assert len(result) == 2
                assert result["Sheet1"] == "model1"
                assert result["Sheet2"] == "model2"

    def test_from_excel_sheet_error(self):
        """Test from_excel when one sheet fails to parse"""
        xlsx_path = Path("/test/data.xlsx")
        generator = ModelGenerator()

        mock_excel = MagicMock()
        mock_excel.sheet_names = ["Sheet1", "BadSheet"]
        mock_excel.parse.side_effect = [
            pd.DataFrame({"col1": [1, 2, 3]}),
            Exception("Sheet parse error")
        ]

        with patch("pandera_forge.generator.ExcelFile") as mock_excel_class:
            mock_excel_class.return_value = mock_excel

            with patch.object(generator, "generate") as mock_generate:
                mock_generate.return_value = "model1"

                with patch("pandera_forge.generator.error") as mock_error:
                    result = generator.from_excel(xlsx_path, validate=False)

                    # Only Sheet1 should be in results
                    assert len(result) == 1
                    assert "Sheet1" in result
                    mock_error.assert_called()

    def test_from_excel_xlrd_error(self):
        """Test from_excel with XLRDError"""
        xlsx_path = Path("/test/data.xlsx")
        generator = ModelGenerator()

        from xlrd import XLRDError

        with patch("pandera_forge.generator.ExcelFile") as mock_excel_class:
            mock_excel_class.side_effect = XLRDError("Unsupported format")

            with patch("pandera_forge.generator.error") as mock_error:
                result = generator.from_excel(xlsx_path, validate=False)

                assert result == {}
                mock_error.assert_called()

    def test_from_excel_single_sheet_error(self):
        """Test from_excel when single sheet fails"""
        xlsx_path = Path("/test/data.xlsx")
        generator = ModelGenerator()

        mock_excel = MagicMock()
        mock_excel.sheet_names = ["Sheet1"]
        mock_excel.parse.side_effect = Exception("Parse error")

        with patch("pandera_forge.generator.ExcelFile") as mock_excel_class:
            mock_excel_class.return_value = mock_excel

            with patch("pandera_forge.generator.error") as mock_error:
                result = generator.from_excel(xlsx_path, validate=False)

                assert result == {}
                mock_error.assert_called()

    def test_generate_implementation_example(self):
        """Test _generate_implementation_example method"""
        generator = ModelGenerator()

        result = generator._generate_implementation_example("TestModel", Path("/data/test.csv"))

        assert "# Example implementation" in result
        assert "TestModel" in result
        assert "test.csv" in result
        assert "read_csv" in result

    def test_generate_field_with_pattern_detection(self):
        """Test _generate_field with pattern detection enabled"""
        df = pd.DataFrame({"email": ["test@example.com", "user@test.org"]})
        generator = ModelGenerator()

        with patch.object(generator.pattern_detector, "detect_pattern") as mock_detect:
            mock_detect.return_value = ("email", r"^[\\w.-]+@[\\w.-]+\\.\\w+$")

            field_str = generator._generate_field(
                df, "email", include_examples=True, detect_patterns=True
            )

            assert field_str is not None
            mock_detect.assert_called_once()

    def test_generate_field_without_pattern_detection(self):
        """Test _generate_field with pattern detection disabled"""
        df = pd.DataFrame({"col1": ["a", "b", "c"]})
        generator = ModelGenerator()

        with patch.object(generator.pattern_detector, "detect_pattern") as mock_detect:
            field_str = generator._generate_field(
                df, "col1", include_examples=True, detect_patterns=False
            )

            assert field_str is not None
            mock_detect.assert_not_called()

    def test_generate_field_no_pandera_type_mapping(self):
        """Test _generate_field when no Pandera type mapping exists"""
        df = pd.DataFrame({"col1": [complex(1, 2), complex(3, 4)]})  # Complex type has no mapping
        generator = ModelGenerator()

        with patch.object(generator.type_mapper, "get_pandera_type") as mock_get_type:
            mock_get_type.return_value = None

            field_str = generator._generate_field(
                df, "col1", include_examples=True, detect_patterns=True
            )

            assert field_str is None