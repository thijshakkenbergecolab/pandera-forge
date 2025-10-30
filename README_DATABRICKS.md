# Databricks Integration for Pandera Forge

This document describes how to use Pandera Forge with Databricks and Apache Spark DataFrames.

## Installation

### For Spark Support
```bash
pip install pandera-forge[spark]
```

### For Full Databricks Support
```bash
pip install pandera-forge[databricks]
```

## Quick Start

### Basic Usage with Databricks

```python
from pandera_forge.databricks import DatabricksGenerator

# Create a generator with Databricks credentials
generator = DatabricksGenerator(
    host="https://your-workspace.cloud.databricks.com",
    token="your-databricks-token",
    cluster_id="your-cluster-id",
    catalog="main",
    schema="default"
)

# Generate model from a table
model_code = generator.from_table(
    table_name="customers",
    detect_patterns=True,
    sample_fraction=0.1  # Sample 10% for analysis
)

print(model_code)
```

### Using with Existing Spark Session

```python
from pyspark.sql import SparkSession
from pandera_forge import ModelGenerator

# Create Spark session
spark = SparkSession.builder.appName("PanderaForge").getOrCreate()

# Create generator for Spark
generator = ModelGenerator.create_for_spark()

# Read DataFrame and generate model
df = spark.table("my_table")
model_code = generator.generate(df, model_name="MyModel")
```

## Configuration Options

### Authentication Methods

#### 1. Direct Credentials
```python
generator = DatabricksGenerator(
    host="https://workspace.cloud.databricks.com",
    token="dapi...",
    cluster_id="0123-456789-abcdef"
)
```

#### 2. Environment Variables
Set these environment variables:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_CLUSTER_ID`
- `DATABRICKS_CATALOG`
- `DATABRICKS_SCHEMA`

```python
# Will use environment variables
generator = DatabricksGenerator()
```

#### 3. Databricks CLI Profile
```python
from pandera_forge.databricks import DatabricksConnector

# Uses ~/.databrickscfg
connector = DatabricksConnector.from_profile("DEFAULT")
generator = DatabricksGenerator(connector=connector)
```

## Features

### Generate Models for Multiple Tables

```python
with DatabricksGenerator(host=host, token=token) as generator:
    # Generate for all tables in schema
    models = generator.generate_for_catalog(
        catalog="analytics",
        schema="sales",
        sample_fraction=0.05
    )

    # Save to directory
    generator.save_models_to_directory(
        models,
        output_dir=Path("models"),
        create_init=True
    )
```

### Delta Lake Support

```python
# Read from Delta table path
model_code = generator.from_delta_path(
    path="/mnt/delta/sales/transactions",
    version=5,  # Optional: specific version
    model_name="TransactionsModel"
)
```

### Unity Catalog Support

```python
generator = DatabricksGenerator(
    host=host,
    token=token,
    catalog="main",  # Unity Catalog
    schema="analytics"
)

# Tables will be read as catalog.schema.table
model_code = generator.from_table("customers")
```

## Performance Considerations

### Sampling
For large tables, use sampling to improve performance:

```python
model_code = generator.from_table(
    table_name="large_table",
    sample_fraction=0.01  # Sample 1% of data
)
```

### Caching
The Spark generator automatically caches DataFrames during analysis for better performance.

### Validation
Note that validation requires converting Spark DataFrames to pandas, which can be expensive for large datasets:

```python
# For large DataFrames, consider disabling validation
model_code = generator.generate(
    df,
    validate=False  # Skip validation for performance
)
```

## Differences from Pandas

### Type Mappings
Spark SQL types are mapped to Pandera types:
- `int`/`integer` → `Int32`
- `bigint`/`long` → `Int64`
- `string`/`varchar` → `String`
- `boolean` → `Bool`
- `timestamp` → `DateTime`
- `array`/`map`/`struct` → `Object`

### Complex Types
Spark's complex types (arrays, maps, structs) are mapped to Pandera's `Object` type:

```python
# Spark DataFrame with complex types
# Column: user_tags (array<string>)
# Generated: user_tags: Object = Field()
```

### Pattern Detection
Pattern detection samples the DataFrame and converts to pandas for analysis:

```python
generator = SparkGenerator(
    sample_size=10000  # Number of rows to sample for pattern detection
)
```

## Example Generated Model

```python
# Generated from Spark DataFrame
# Note: Validation with Spark DataFrames requires conversion to pandas

from pandera import DataFrameModel, Field
from pandera.typing import Series
from pandera.typing.pandas import Int64, String, Float64, DateTime

class SalesDataModel(DataFrameModel):
    """Pandera DataFrameModel for SalesData"""

    transaction_id: Series[Int64] = Field(unique=True)
    customer_id: Series[Int64] = Field()
    product_name: Series[String] = Field()
    amount: Series[Float64] = Field(ge=0.0, le=10000.0)
    transaction_date: Series[DateTime] = Field()

    class Config:
        strict = True
```

## Troubleshooting

### Import Errors
If you get import errors, ensure you have the required dependencies:
```bash
pip install pyspark databricks-sdk databricks-sql-connector
```

### Connection Issues
- Verify your Databricks host URL (should start with `https://`)
- Check that your token is valid and has necessary permissions
- Ensure the cluster is running and accessible

### Memory Issues
For large DataFrames:
- Use sampling (`sample_fraction` parameter)
- Disable validation (`validate=False`)
- Increase Spark executor memory if needed

### Type Mapping Issues
If a Spark type isn't recognized, it will be skipped with a warning. You can extend the `SparkTypeMapper` class to add custom mappings.

## Advanced Usage

### Custom Type Mappings

```python
from pandera_forge.spark import SparkTypeMapper

class CustomSparkTypeMapper(SparkTypeMapper):
    def __init__(self):
        super().__init__()
        # Add custom mappings
        self.SPARK_TO_PANDERA_MAP["custom_type"] = CustomPanderaType
```

### Integration with MLflow

```python
import mlflow
from pandera_forge.databricks import DatabricksGenerator

# Generate model
generator = DatabricksGenerator(...)
model_code = generator.from_table("features")

# Log with MLflow
with mlflow.start_run():
    mlflow.log_text(model_code, "schema/model.py")
```

## Limitations

1. **Validation**: Full DataFrame validation requires converting Spark DataFrames to pandas, which may not be feasible for very large datasets.

2. **Pattern Detection**: Pattern detection uses sampling and pandas conversion, so patterns might be missed in very large datasets with rare patterns.

3. **Complex Types**: Nested structures (arrays, maps, structs) are mapped to generic `Object` type without detailed schema.

4. **Streaming DataFrames**: Not currently supported. Only batch DataFrames are supported.

## Contributing

We welcome contributions! Areas for improvement:
- Better complex type handling
- Streaming DataFrame support
- Native Spark validation without pandas conversion
- Additional pattern detection algorithms for distributed data