"""
Abstract base class for DataFrame model generation
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Dict

from ..code_generator import CodeGenerator
from ..name_sanitizer import NameSanitizer
from ..pattern_detector import PatternDetector
from ..validator import ModelValidator
from ..llm_enricher import LLMEnricher


class BaseGenerator(ABC):
    """Abstract base class for generating Pandera models from DataFrames"""

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_enricher: Optional[LLMEnricher] = None
    ):
        # Common components that work across implementations
        self.name_sanitizer = NameSanitizer()
        self.code_generator = CodeGenerator()
        self.validator = ModelValidator()
        self.pattern_detector = PatternDetector()

        # LLM enricher is optional and common
        if llm_enricher:
            self.llm_enricher = llm_enricher
        else:
            self.llm_enricher = LLMEnricher(api_key=llm_api_key) if llm_api_key else None

        # Implementation-specific components will be set by subclasses
        self.type_mapper = None
        self.field_analyzer = None

    @abstractmethod
    def generate(
        self,
        df: Any,
        model_name: str = "DataFrameModel",
        validate: bool = True,
        include_examples: bool = True,
        detect_patterns: bool = True,
        source_file: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a DataFrame.

        Args:
            df: Source DataFrame (pandas, Spark, etc.)
            model_name: Name for the generated model class
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
            source_file: Optional path to source file for implementation example

        Returns:
            Generated model code as string, or None if generation failed
        """
        pass

    @abstractmethod
    def get_columns(self, df: Any) -> list:
        """
        Get list of column names from the DataFrame.

        Args:
            df: The DataFrame

        Returns:
            List of column names
        """
        pass

    @abstractmethod
    def get_column_data(self, df: Any, column: str) -> Any:
        """
        Get data from a specific column.

        Args:
            df: The DataFrame
            column: Column name

        Returns:
            Column data in a format suitable for pattern detection
        """
        pass

    @abstractmethod
    def validate_dataframe(self, df: Any, model_code: str, model_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate the generated model against the source DataFrame.

        Args:
            df: The DataFrame to validate
            model_code: Generated model code
            model_name: Name of the model class

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def _generate_field(
        self,
        df: Any,
        column: str,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate field definition for a single column.
        This method can be overridden by subclasses if needed.

        Args:
            df: The DataFrame
            column: Column name
            include_examples: Whether to include examples
            detect_patterns: Whether to detect patterns

        Returns:
            Field definition string or None
        """
        # Get dtype and Pandera type
        dtype_str = self.field_analyzer.get_column_dtype(df, column)
        pandera_type = self.type_mapper.get_pandera_type(dtype_str)

        if not pandera_type:
            print(f"Warning: No Pandera type mapping for dtype {dtype_str} in column {column}")
            return None

        # Sanitize column name
        sanitized_name, is_valid_name = self.name_sanitizer.sanitize_column_name(column)

        # Analyze column properties
        properties = self.field_analyzer.analyze_column(df, column)

        # Detect patterns for string columns if enabled
        if detect_patterns and pandera_type.__name__ in ["String", "Object"]:
            column_data = self.get_column_data(df, column)
            pattern_result = self.pattern_detector.detect_pattern(column_data)
            if pattern_result:
                properties["pattern_name"] = pattern_result[0]
                properties["pattern"] = pattern_result[1]

            # Get string constraints
            string_constraints = self.pattern_detector.infer_string_constraints(column_data)
            properties.update(string_constraints)

        # Generate field string
        field_str = self.code_generator.generate_field_string(
            field_name=sanitized_name,
            pandera_type=pandera_type.__name__,
            properties=properties,
            original_column_name=column,
            needs_alias=not is_valid_name,
        )

        # Add pattern comment if detected
        if properties.get("pattern_name"):
            field_str += f"  # pattern: {properties['pattern_name']}"

        return field_str

    def _generate_implementation_example(self, model_name: str, source_file: Path) -> str:
        """
        Generate example implementation code.
        This is common across implementations.

        Args:
            model_name: Name of the model class
            source_file: Path to the source file

        Returns:
            Implementation example code
        """
        return f"""# Example implementation


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    # Load the data
    file_path = Path("{source_file.absolute()}")

    # Read file based on extension
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {source_file.suffix}")

    # Validate the DataFrame
    validated_df = {model_name}.validate(df)
    print(f"Successfully validated {{len(validated_df)}} rows")
"""