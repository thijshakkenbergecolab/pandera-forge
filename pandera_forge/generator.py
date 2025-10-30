"""
Main generator class that orchestrates the model generation process
"""

from logging import error
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pandas import DataFrame, read_csv, ExcelFile
from xlrd import XLRDError

from .code_generator import CodeGenerator
from .field_analyzer import FieldAnalyzer
from .llm_enricher import LLMEnricher
from .name_sanitizer import NameSanitizer
from .pattern_detector import PatternDetector
from .type_mapper import TypeMapper
from .validator import ModelValidator
from .pandas.pandas_generator import PandasGenerator


class ModelGenerator:
    """
    Main class for generating Pandera models from DataFrames.

    This class now acts as a factory that delegates to the appropriate
    implementation based on the DataFrame type (pandas or Spark).
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_enricher: Optional[LLMEnricher] = None,
        backend: str = "pandas"
    ):
        """
        Initialize the ModelGenerator.

        Args:
            llm_api_key: Optional API key for LLM enrichment
            llm_enricher: Optional pre-configured LLM enricher
            backend: Backend to use ("pandas", "spark", or "auto")
        """
        self.llm_api_key = llm_api_key
        self.llm_enricher = llm_enricher
        self.backend = backend

        # For backward compatibility, keep pandas components as defaults
        self._pandas_generator = PandasGenerator(llm_api_key, llm_enricher)
        self._spark_generator = None

        # Legacy attributes for backward compatibility
        self.type_mapper = self._pandas_generator.type_mapper
        self.name_sanitizer = self._pandas_generator.name_sanitizer
        self.field_analyzer = self._pandas_generator.field_analyzer
        self.code_generator = self._pandas_generator.code_generator
        self.validator = self._pandas_generator.validator
        self.pattern_detector = self._pandas_generator.pattern_detector

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

        Automatically detects the DataFrame type and uses the appropriate generator.

        Args:
            df: Source DataFrame (pandas or Spark)
            model_name: Name for the generated model class
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
            source_file: Optional path to source file for implementation example

        Returns:
            Generated model code as string, or None if generation failed
        """
        # Detect DataFrame type and delegate to appropriate generator
        generator = self._get_generator_for_dataframe(df)

        return generator.generate(
            df=df,
            model_name=model_name,
            validate=validate,
            include_examples=include_examples,
            detect_patterns=detect_patterns,
            source_file=source_file
        )

    def _get_generator_for_dataframe(self, df: Any):
        """
        Get the appropriate generator based on DataFrame type.

        Args:
            df: The DataFrame to analyze

        Returns:
            Appropriate generator instance
        """
        # Check if it's a Spark DataFrame
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            if isinstance(df, SparkDataFrame):
                if self._spark_generator is None:
                    from .spark.spark_generator import SparkGenerator
                    self._spark_generator = SparkGenerator(
                        self.llm_api_key, self.llm_enricher
                    )
                return self._spark_generator
        except ImportError:
            pass

        # Check if it's a pandas DataFrame
        if isinstance(df, DataFrame):
            return self._pandas_generator

        # If backend is explicitly set to spark, try to use Spark generator
        if self.backend == "spark":
            if self._spark_generator is None:
                from .spark.spark_generator import SparkGenerator
                self._spark_generator = SparkGenerator(
                    self.llm_api_key, self.llm_enricher
                )
            return self._spark_generator

        # Default to pandas generator
        return self._pandas_generator

    def from_csv(
        self,
        csv_path: Path,
        validate: bool = True,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate a Pandera DataFrameModel from a CSV file.

        For backward compatibility, this delegates to the pandas generator.

        Args:
            csv_path: Path to the CSV file
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
        Returns:
            Generated model code as string, or None if generation failed
        """
        return self._pandas_generator.from_csv(
            csv_path=csv_path,
            validate=validate,
            include_examples=include_examples,
            detect_patterns=detect_patterns
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

        For backward compatibility, this delegates to the pandas generator.

        Args:
            xlsx_path: Path to the Excel file
            validate: Whether to validate the generated model
            include_examples: Whether to include example values in comments
            detect_patterns: Whether to detect patterns in string columns
        Returns:
            Dictionary mapping sheet names to generated model code
        """
        return self._pandas_generator.from_excel(
            xlsx_path=xlsx_path,
            validate=validate,
            include_examples=include_examples,
            detect_patterns=detect_patterns
        )

    def _generate_field(
        self,
        df: DataFrame,
        column: str,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate field definition for a single column.

        For backward compatibility, delegates to pandas generator.
        """
        return self._pandas_generator._generate_field(
            df, column, include_examples, detect_patterns
        )

    def _generate_implementation_example(self, model_name: str, source_file: Path) -> str:
        """
        Generate example implementation code.

        For backward compatibility, delegates to pandas generator.
        """
        return self._pandas_generator._generate_implementation_example(
            model_name, source_file
        )

    @classmethod
    def create_for_spark(cls, llm_api_key: Optional[str] = None, sample_size: Optional[int] = 10000):
        """
        Create a ModelGenerator configured for Spark DataFrames.

        Args:
            llm_api_key: Optional API key for LLM enrichment
            sample_size: Number of rows to sample for analysis

        Returns:
            ModelGenerator instance configured for Spark
        """
        from .spark.spark_generator import SparkGenerator

        generator = cls(llm_api_key=llm_api_key, backend="spark")
        generator._spark_generator = SparkGenerator(
            llm_api_key=llm_api_key,
            sample_size=sample_size
        )
        return generator

    @classmethod
    def create_for_databricks(
        cls,
        host: Optional[str] = None,
        token: Optional[str] = None,
        cluster_id: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        sample_size: Optional[int] = 10000,
    ):
        """
        Create a ModelGenerator configured for Databricks.

        Args:
            host: Databricks workspace URL
            token: Databricks personal access token
            cluster_id: ID of the compute cluster
            catalog: Unity Catalog name
            schema: Schema/database name
            llm_api_key: Optional API key for LLM enrichment
            sample_size: Number of rows to sample for analysis

        Returns:
            DatabricksGenerator instance
        """
        from .databricks.generator import DatabricksGenerator

        return DatabricksGenerator(
            host=host,
            token=token,
            cluster_id=cluster_id,
            catalog=catalog,
            schema=schema,
            llm_api_key=llm_api_key,
            sample_size=sample_size
        )
