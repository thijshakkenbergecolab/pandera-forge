"""
Spark-specific generator implementation
"""

from pathlib import Path
from typing import Optional, Any, List

from ..core.base_generator import BaseGenerator
from .spark_type_mapper import SparkTypeMapper
from .spark_field_analyzer import SparkFieldAnalyzer


class SparkGenerator(BaseGenerator):
    """Spark-specific implementation for generating Pandera models from Spark DataFrames"""

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_enricher: Optional[Any] = None,
        sample_size: Optional[int] = 10000
    ):
        """
        Initialize the Spark generator.

        Args:
            llm_api_key: Optional API key for LLM enrichment
            llm_enricher: Optional pre-configured LLM enricher
            sample_size: Number of rows to sample for analysis (None for full DataFrame)
        """
        super().__init__(llm_api_key, llm_enricher)
        self.type_mapper = SparkTypeMapper()
        self.field_analyzer = SparkFieldAnalyzer(sample_size=sample_size)
        self.sample_size = sample_size

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
        Generate a Pandera DataFrameModel from a Spark DataFrame.

        Args:
            df: Source Spark DataFrame
            model_name: Name for the generated model class
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
            source_file: Optional path to source file for implementation example

        Returns:
            Generated model code as string, or None if generation failed
        """
        try:
            # Check if PySpark is available
            from pyspark.sql import DataFrame as SparkDataFrame
            if not isinstance(df, SparkDataFrame):
                print(f"Warning: Expected Spark DataFrame, got {type(df)}")
                return None
        except ImportError:
            print("Warning: PySpark not installed. Install with: pip install pyspark")
            return None

        # Sanitize the model name
        sanitized_model_name = self.name_sanitizer.sanitize_class_name(model_name)

        # Cache the DataFrame for better performance during analysis
        df.cache()

        # Generate fields for each column
        fields = []
        for column in self.get_columns(df):
            field_code = self._generate_field(df, column, include_examples, detect_patterns)
            if field_code:
                fields.append(field_code)

        # Unpersist the cached DataFrame
        df.unpersist()

        if not fields:
            print(f"Warning: No valid fields generated for model {sanitized_model_name}")
            return None

        # Generate the class definition
        class_code = self.code_generator.generate_class_definition(sanitized_model_name, fields)

        # Generate complete code with imports
        full_code = self.code_generator.generate_imports() + "\n\n\n" + class_code

        # Add Spark-specific comment
        full_code = (
            "# Generated from Spark DataFrame\n"
            "# Note: Validation with Spark DataFrames requires conversion to pandas\n\n" +
            full_code
        )

        # Validate if requested (requires conversion to pandas)
        if validate:
            is_valid, error_msg = self.validator.validate_model_code(full_code, sanitized_model_name)
            if not is_valid:
                print(f"Warning: Generated model has syntax errors: {error_msg}")
                return None

            # Skip DataFrame validation for Spark (would require full collect)
            print("Note: Skipping DataFrame validation for Spark (requires full collect)")

        # Add implementation example if source file provided
        if source_file:
            implementation = self._generate_spark_implementation_example(
                sanitized_model_name, source_file
            )
            full_code += "\n\n" + implementation

        return full_code

    def get_columns(self, df: Any) -> List[str]:
        """Get list of column names from Spark DataFrame."""
        return df.columns

    def get_column_data(self, df: Any, column: str) -> Any:
        """
        Get data from a specific Spark column for pattern detection.
        Returns a pandas Series for compatibility with pattern detector.
        """
        try:
            # Sample and convert to pandas for pattern detection
            sample_size = min(self.sample_size or 10000, df.count())
            fraction = sample_size / df.count()

            # Sample and select the column
            sampled_df = df.sample(withReplacement=False, fraction=fraction)
            pandas_df = sampled_df.select(column).toPandas()
            return pandas_df[column]
        except:
            return None

    def validate_dataframe(self, df: Any, model_code: str, model_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate the generated model against a Spark DataFrame.
        Note: This requires converting the Spark DataFrame to pandas.
        """
        try:
            # Convert Spark DataFrame to pandas for validation
            # This is expensive for large DataFrames
            pandas_df = df.toPandas()
            return self.validator.validate_against_dataframe(model_code, model_name, pandas_df)
        except Exception as e:
            return False, f"Cannot validate Spark DataFrame: {e}"

    def from_table(
        self,
        table_name: str,
        spark_session: Any,
        model_name: Optional[str] = None,
        validate: bool = False,  # Default to False for Spark
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a Spark table.

        Args:
            table_name: Name of the Spark table
            spark_session: Active SparkSession
            model_name: Name for the generated model class (defaults to table name)
            validate: Whether to validate the generated model (requires conversion to pandas)
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns

        Returns:
            Generated model code as string, or None if generation failed
        """
        try:
            # Read the table into a Spark DataFrame
            df = spark_session.table(table_name)

            # Use table name as model name if not provided
            if model_name is None:
                model_name = table_name.replace(".", "_")

            return self.generate(
                df,
                model_name=model_name,
                validate=validate,
                include_examples=include_examples,
                detect_patterns=detect_patterns,
            )
        except Exception as e:
            print(f"Error reading Spark table {table_name}: {e}")
            return None

    def from_parquet(
        self,
        parquet_path: str,
        spark_session: Any,
        model_name: Optional[str] = None,
        validate: bool = False,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a Parquet file using Spark.

        Args:
            parquet_path: Path to the Parquet file(s)
            spark_session: Active SparkSession
            model_name: Name for the generated model class
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns

        Returns:
            Generated model code as string, or None if generation failed
        """
        try:
            # Read the Parquet file into a Spark DataFrame
            df = spark_session.read.parquet(parquet_path)

            # Use file name as model name if not provided
            if model_name is None:
                model_name = Path(parquet_path).stem.replace(" ", "_").replace("-", "_")

            return self.generate(
                df,
                model_name=model_name,
                validate=validate,
                include_examples=include_examples,
                detect_patterns=detect_patterns,
                source_file=Path(parquet_path),
            )
        except Exception as e:
            print(f"Error reading Parquet file {parquet_path}: {e}")
            return None

    def _generate_spark_implementation_example(self, model_name: str, source_file: Path) -> str:
        """
        Generate Spark-specific example implementation code.

        Args:
            model_name: Name of the model class
            source_file: Path to the source file

        Returns:
            Implementation example code
        """
        return f"""# Example implementation for Spark


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pathlib import Path

    # Initialize Spark session
    spark = SparkSession.builder \\
        .appName("Pandera Validation") \\
        .getOrCreate()

    # Load the data
    file_path = "{source_file.absolute()}"

    # Read file based on extension
    if file_path.endswith(".parquet"):
        df = spark.read.parquet(file_path)
    elif file_path.endswith(".csv"):
        df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif file_path.endswith(".json"):
        df = spark.read.json(file_path)
    else:
        # Assume it's a table name
        df = spark.table(file_path)

    # For validation, we need to convert to pandas
    # Note: This can be expensive for large DataFrames
    # Consider sampling for validation
    sample_size = 10000
    if df.count() > sample_size:
        print(f"Sampling {{sample_size}} rows for validation...")
        pandas_df = df.sample(withReplacement=False, fraction=sample_size/df.count()).toPandas()
    else:
        pandas_df = df.toPandas()

    # Validate the DataFrame
    validated_df = {model_name}.validate(pandas_df)
    print(f"Successfully validated {{len(validated_df)}} rows")

    # Stop Spark session
    spark.stop()
"""