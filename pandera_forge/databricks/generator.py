"""
Databricks-specific generator that combines Spark generator with Databricks connectivity
"""

from typing import Optional, Any, Dict, List
from pathlib import Path

from ..spark.spark_generator import SparkGenerator
from .connector import DatabricksConnector


class DatabricksGenerator(SparkGenerator):
    """Databricks-specific generator with built-in connectivity"""

    def __init__(
        self,
        connector: Optional[DatabricksConnector] = None,
        host: Optional[str] = None,
        token: Optional[str] = None,
        cluster_id: Optional[str] = None,
        sql_endpoint_id: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_enricher: Optional[Any] = None,
        sample_size: Optional[int] = 10000,
    ):
        """
        Initialize Databricks generator.

        Args:
            connector: Pre-configured DatabricksConnector instance
            host: Databricks workspace URL
            token: Databricks personal access token
            cluster_id: ID of the compute cluster
            sql_endpoint_id: ID of the SQL warehouse endpoint
            catalog: Unity Catalog name
            schema: Schema/database name
            llm_api_key: Optional API key for LLM enrichment
            llm_enricher: Optional pre-configured LLM enricher
            sample_size: Number of rows to sample for analysis
        """
        super().__init__(llm_api_key, llm_enricher, sample_size)

        # Use provided connector or create a new one
        if connector:
            self.connector = connector
        else:
            self.connector = DatabricksConnector(
                host=host,
                token=token,
                cluster_id=cluster_id,
                sql_endpoint_id=sql_endpoint_id,
                catalog=catalog,
                schema=schema,
            )

        # Get Spark session from connector
        self.spark = self.connector.get_spark_session()

    def from_table(
        self,
        table_name: str,
        model_name: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        validate: bool = False,
        include_examples: bool = True,
        detect_patterns: bool = True,
        sample_fraction: Optional[float] = None,
    ) -> Optional[str]:
        """
        Generate a Pandera model from a Databricks table.

        Args:
            table_name: Name of the table
            model_name: Name for the generated model class
            catalog: Catalog name (uses connector default if not provided)
            schema: Schema name (uses connector default if not provided)
            validate: Whether to validate the generated model
            include_examples: Whether to include example values
            detect_patterns: Whether to detect patterns in string columns
            sample_fraction: Optional fraction to sample (0.0 to 1.0)

        Returns:
            Generated model code as string
        """
        # Read the table
        df = self.connector.read_table(
            table_name=table_name,
            catalog=catalog,
            schema=schema,
            sample_fraction=sample_fraction
        )

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

    def from_delta_path(
        self,
        path: str,
        model_name: Optional[str] = None,
        version: Optional[int] = None,
        validate: bool = False,
        include_examples: bool = True,
        detect_patterns: bool = True,
    ) -> Optional[str]:
        """
        Generate a Pandera model from a Delta table path.

        Args:
            path: Path to the Delta table
            model_name: Name for the generated model class
            version: Optional Delta table version to read
            validate: Whether to validate the generated model
            include_examples: Whether to include example values
            detect_patterns: Whether to detect patterns in string columns

        Returns:
            Generated model code as string
        """
        # Read the Delta table
        df = self.connector.read_delta_table(path, version)

        # Use path as model name if not provided
        if model_name is None:
            model_name = Path(path).name.replace("-", "_").replace(" ", "_")

        return self.generate(
            df,
            model_name=model_name,
            validate=validate,
            include_examples=include_examples,
            detect_patterns=detect_patterns,
        )

    def generate_for_catalog(
        self,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        tables: Optional[List[str]] = None,
        validate: bool = False,
        include_examples: bool = True,
        detect_patterns: bool = True,
        sample_fraction: Optional[float] = 0.1,
    ) -> Dict[str, str]:
        """
        Generate Pandera models for multiple tables in a catalog/schema.

        Args:
            catalog: Catalog name (uses connector default if not provided)
            schema: Schema name (uses connector default if not provided)
            tables: List of specific tables (generates for all if not provided)
            validate: Whether to validate the generated models
            include_examples: Whether to include example values
            detect_patterns: Whether to detect patterns in string columns
            sample_fraction: Fraction to sample from each table

        Returns:
            Dictionary mapping table names to generated model code
        """
        models = {}

        # Get list of tables
        if tables is None:
            tables = self.connector.list_tables(catalog, schema)

        print(f"Generating models for {len(tables)} tables...")

        for table_name in tables:
            try:
                print(f"Processing table: {table_name}")
                model_code = self.from_table(
                    table_name=table_name,
                    catalog=catalog,
                    schema=schema,
                    validate=validate,
                    include_examples=include_examples,
                    detect_patterns=detect_patterns,
                    sample_fraction=sample_fraction,
                )

                if model_code:
                    models[table_name] = model_code
                    print(f"  ✓ Generated model for {table_name}")
                else:
                    print(f"  ✗ Failed to generate model for {table_name}")

            except Exception as e:
                print(f"  ✗ Error processing {table_name}: {e}")

        return models

    def save_models_to_directory(
        self,
        models: Dict[str, str],
        output_dir: Path,
        create_init: bool = True,
    ):
        """
        Save generated models to a directory.

        Args:
            models: Dictionary mapping table names to model code
            output_dir: Directory to save the models
            create_init: Whether to create an __init__.py file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each model to a separate file
        for table_name, model_code in models.items():
            file_name = f"{table_name.lower()}_model.py"
            file_path = output_dir / file_name

            with open(file_path, "w") as f:
                f.write(model_code)

            print(f"Saved model to {file_path}")

        # Create __init__.py if requested
        if create_init:
            init_path = output_dir / "__init__.py"
            init_content = '"""Generated Pandera models for Databricks tables"""\n\n'

            # Add imports for all models
            for table_name in models.keys():
                module_name = f"{table_name.lower()}_model"
                class_name = table_name.replace("_", "")
                init_content += f"from .{module_name} import {class_name}\n"

            init_content += "\n__all__ = [\n"
            for table_name in models.keys():
                class_name = table_name.replace("_", "")
                init_content += f'    "{class_name}",\n'
            init_content += "]\n"

            with open(init_path, "w") as f:
                f.write(init_content)

            print(f"Created {init_path}")

    def close(self):
        """Close the Databricks connection"""
        self.connector.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()