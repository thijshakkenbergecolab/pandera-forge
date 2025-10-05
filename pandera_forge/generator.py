"""
Main generator class that orchestrates the model generation process
"""

from pathlib import Path
from typing import Optional
from pandas import DataFrame

from .code_generator import CodeGenerator
from .field_analyzer import FieldAnalyzer
from .llm_enricher import LLMEnricher
from .name_sanitizer import NameSanitizer
from .pattern_detector import PatternDetector
from .type_mapper import TypeMapper
from .validator import ModelValidator


class ModelGenerator:
    """Main class for generating Pandera models from DataFrames"""

    def __init__(
        self, llm_api_key: Optional[str] = None, llm_enricher: Optional[LLMEnricher] = None
    ):
        self.type_mapper = TypeMapper()
        self.name_sanitizer = NameSanitizer()
        self.field_analyzer = FieldAnalyzer()
        self.code_generator = CodeGenerator()
        self.validator = ModelValidator()
        self.pattern_detector = PatternDetector()
        # Allow passing a pre-configured LLMEnricher or create one with API key
        if llm_enricher:
            self.llm_enricher = llm_enricher
        else:
            self.llm_enricher = LLMEnricher(api_key=llm_api_key)

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
            df: Source DataFrame
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
        for column in df.columns:
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
            is_valid, error = self.validator.validate_model_code(full_code, sanitized_model_name)
            if not is_valid:
                print(f"Warning: Generated model has syntax errors: {error}")
                return None

            # Validate against DataFrame
            is_valid, error = self.validator.validate_against_dataframe(
                class_code, sanitized_model_name, df
            )
            if not is_valid:
                print(f"Warning: Model validation against DataFrame failed: {error}")
                return None

        # Add implementation example if source file provided
        if source_file:
            implementation = self._generate_implementation_example(
                sanitized_model_name, source_file
            )
            full_code += "\n\n" + implementation

        return full_code

    def _generate_field(
        self,
        df: DataFrame,
        column: str,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """Generate field definition for a single column"""
        # Get pandera type
        dtype_str = str(df[column].dtype)
        # Handle timestamp types
        if "datetime64" in dtype_str:
            dtype_str = "datetime64[ns]"
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
            pattern_result = self.pattern_detector.detect_pattern(df[column])
            if pattern_result:
                properties["pattern_name"] = pattern_result[0]
                properties["pattern"] = pattern_result[1]

            # Get string constraints
            string_constraints = self.pattern_detector.infer_string_constraints(df[column])
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
        """Generate example implementation code"""
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
