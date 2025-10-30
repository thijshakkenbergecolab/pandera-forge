"""
Example usage of Pandera Forge with Databricks
"""

from pathlib import Path
from pandera_forge import ModelGenerator
from pandera_forge.databricks import DatabricksGenerator, DatabricksConnector


def example_basic_databricks():
    """Basic example using Databricks tables"""

    # Create a Databricks generator
    generator = DatabricksGenerator(
        host="https://your-workspace.cloud.databricks.com",
        token="your-databricks-token",
        cluster_id="your-cluster-id",
        catalog="main",  # Unity Catalog name
        schema="default"
    )

    # Generate model from a single table
    model_code = generator.from_table(
        table_name="customers",
        validate=False,  # Validation requires converting to pandas
        include_examples=True,
        detect_patterns=True,
        sample_fraction=0.1  # Sample 10% of data for analysis
    )

    if model_code:
        # Save to file
        with open("customers_model.py", "w") as f:
            f.write(model_code)
        print("Model generated successfully!")


def example_multiple_tables():
    """Generate models for multiple tables in a catalog"""

    with DatabricksGenerator(
        host="https://your-workspace.cloud.databricks.com",
        token="your-databricks-token",
        catalog="analytics",
        schema="sales"
    ) as generator:

        # Generate models for all tables in the schema
        models = generator.generate_for_catalog(
            validate=False,
            include_examples=True,
            detect_patterns=True,
            sample_fraction=0.05  # Sample 5% for faster processing
        )

        # Save all models to a directory
        generator.save_models_to_directory(
            models,
            output_dir=Path("generated_models"),
            create_init=True
        )


def example_delta_table():
    """Generate model from a Delta table path"""

    generator = DatabricksGenerator(
        host="https://your-workspace.cloud.databricks.com",
        token="your-databricks-token"
    )

    # Generate from Delta table path
    model_code = generator.from_delta_path(
        path="/mnt/delta/sales/transactions",
        model_name="TransactionsModel",
        version=5,  # Optional: specific Delta version
        validate=False,
        detect_patterns=True
    )

    print(model_code)


def example_with_spark_session():
    """Use with an existing Spark session"""

    from pyspark.sql import SparkSession

    # Create or get existing Spark session
    spark = SparkSession.builder \
        .appName("PanderaForge") \
        .config("spark.databricks.service.address", "https://your-workspace.cloud.databricks.com") \
        .config("spark.databricks.service.token", "your-token") \
        .getOrCreate()

    # Use the factory method for Spark DataFrames
    generator = ModelGenerator.create_for_spark()

    # Read data and generate model
    df = spark.table("main.default.products")
    model_code = generator.generate(
        df,
        model_name="ProductsModel",
        validate=False,  # Validation requires pandas conversion
        detect_patterns=True
    )

    print(model_code)
    spark.stop()


def example_from_profile():
    """Use Databricks CLI profile for authentication"""

    # Uses ~/.databrickscfg file
    connector = DatabricksConnector.from_profile("DEFAULT")

    generator = DatabricksGenerator(
        connector=connector,
        catalog="main",
        schema="analytics"
    )

    # Generate model
    model_code = generator.from_table(
        table_name="user_events",
        detect_patterns=True
    )

    print(model_code)


def example_with_llm_enrichment():
    """Example with LLM pattern detection"""

    generator = DatabricksGenerator(
        host="https://your-workspace.cloud.databricks.com",
        token="your-databricks-token",
        llm_api_key="your-openai-api-key",  # For enhanced pattern detection
        sample_size=5000  # Sample size for analysis
    )

    model_code = generator.from_table(
        table_name="customer_data",
        detect_patterns=True,  # Will use LLM for better pattern detection
        include_examples=True
    )

    print(model_code)


if __name__ == "__main__":
    # Note: Replace with your actual Databricks credentials
    print("Databricks integration examples")
    print("=" * 50)
    print("1. Basic table generation")
    print("2. Multiple tables from catalog")
    print("3. Delta table from path")
    print("4. Using existing Spark session")
    print("5. Using CLI profile")
    print("6. With LLM enrichment")
    print("\nUpdate the credentials in the examples before running!")