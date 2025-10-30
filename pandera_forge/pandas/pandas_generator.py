"""
Pandas-specific generator implementation
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pandas import DataFrame, read_csv, ExcelFile
from xlrd import XLRDError
from logging import error

from ..core.base_generator import BaseGenerator
from .pandas_type_mapper import PandasTypeMapper
from .pandas_field_analyzer import PandasFieldAnalyzer


class PandasGenerator(BaseGenerator):
    """Pandas-specific implementation for generating Pandera models from pandas DataFrames"""

    def __init__(self, llm_api_key: Optional[str] = None, llm_enricher: Optional[Any] = None):
        super().__init__(llm_api_key, llm_enricher)
        self.type_mapper = PandasTypeMapper()
        self.field_analyzer = PandasFieldAnalyzer()

    def generate(
        self,
        df: DataFrame,
        model_name: str = "DataFrameModel",
        validate: bool = True,
        include_examples: bool = True,
        detect_patterns: bool = True,
        source_file: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a pandas DataFrame.

        Args:
            df: Source pandas DataFrame
            model_name: Name for the generated model class
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
            source_file: Optional path to source file for implementation example

        Returns:
            Generated model code as string, or None if generation failed
        """
        # Sanitize the model name
        sanitized_model_name = self.name_sanitizer.sanitize_class_name(model_name)

        # Generate fields for each column
        fields = []
        for column in self.get_columns(df):
            field_code = self._generate_field(df, column, include_examples, detect_patterns)
            if field_code:
                fields.append(field_code)

        if not fields:
            print(f"Warning: No valid fields generated for model {sanitized_model_name}")
            return None

        # Generate the class definition
        class_code = self.code_generator.generate_class_definition(sanitized_model_name, fields)

        # Generate complete code with imports
        full_code = self.code_generator.generate_imports() + "\n\n\n" + class_code

        # Validate if requested
        if validate:
            is_valid, error_msg = self.validator.validate_model_code(full_code, sanitized_model_name)
            if not is_valid:
                print(f"Warning: Generated model has syntax errors: {error_msg}")
                return None

            # Validate against DataFrame
            is_valid, error_msg = self.validate_dataframe(df, class_code, sanitized_model_name)
            if not is_valid:
                print(f"Warning: Model validation against DataFrame failed: {error_msg}")
                return None

        # Add implementation example if source file provided
        if source_file:
            implementation = self._generate_implementation_example(
                sanitized_model_name, source_file
            )
            full_code += "\n\n" + implementation

        return full_code

    def get_columns(self, df: DataFrame) -> list:
        """Get list of column names from pandas DataFrame."""
        return df.columns.tolist()

    def get_column_data(self, df: DataFrame, column: str) -> Any:
        """Get pandas Series for a column."""
        return df[column]

    def validate_dataframe(self, df: DataFrame, model_code: str, model_name: str) -> tuple[bool, Optional[str]]:
        """Validate the generated model against a pandas DataFrame."""
        return self.validator.validate_against_dataframe(model_code, model_name, df)

    def from_csv(
        self,
        csv_path: Path,
        validate: bool = True,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a CSV file.

        Args:
            csv_path: Path to the CSV file
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
        Returns:
            Generated model code as string, or None if generation failed
        """
        # Read the CSV file into a DataFrame
        try:
            df = read_csv(csv_path)
        except UnicodeError as ue:
            # encoding='latin-1
            try:
                print(f"UnicodeError reading {csv_path}, trying latin-1 encoding: {ue}")
                df = read_csv(csv_path, encoding="latin-1")  # type: ignore
            except Exception as e:
                print(f"Error reading CSV file {csv_path} with latin-1 encoding: {e}")
                return None
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}")
            return None

        model_name = csv_path.stem.replace(" ", "_").replace("-", "_")
        return self.generate(
            df,
            model_name=model_name,
            validate=validate,
            include_examples=include_examples,
            detect_patterns=detect_patterns,
            source_file=csv_path,
        )

    def from_excel(
        self,
        xlsx_path: Path,
        validate: bool = True,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Dict[str, str]:
        """
        Generate Pandera DataFrameModels from an Excel file.

        Args:
            xlsx_path: Path to the Excel file
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
        Returns:
            Dictionary mapping sheet names to generated model code
        """
        # Read the Excel file into a DataFrame / set of dataframes
        models = {}
        try:
            file = ExcelFile(xlsx_path)
        except XLRDError as xe:
            error(f"XLRDError reading {xlsx_path}, it might be an unsupported format: {xe}")
            return models

        if len(file.sheet_names) > 1:
            for sheet in file.sheet_names:
                try:
                    df = file.parse(sheet_name=sheet)
                    print(f"Sheet: {sheet}, Rows: {len(df)}, Columns: {len(df.columns)}")
                    model_name = f"{xlsx_path.stem}_{sheet}".replace(" ", "_").replace("-", "_")
                    model_code = self.generate(
                        df,
                        model_name=model_name,
                        validate=validate,
                        include_examples=include_examples,
                        detect_patterns=detect_patterns,
                        source_file=xlsx_path,
                    )
                    if model_code:
                        models[sheet] = model_code
                except Exception as e:
                    error(f"Error reading sheet {sheet} in {xlsx_path}: {e}")
        else:
            try:
                df = file.parse(sheet_name=file.sheet_names[0])
                model_name = xlsx_path.stem.replace(" ", "_").replace("-", "_")
                model_code = self.generate(
                    df,
                    model_name=model_name,
                    validate=validate,
                    include_examples=include_examples,
                    detect_patterns=detect_patterns,
                    source_file=xlsx_path,
                )
                if model_code:
                    models[xlsx_path.stem] = model_code
            except Exception as e:
                error(f"Error reading Excel file {xlsx_path}: {e}")

        return models