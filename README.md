# Pandera Forge 🔨

**Pandera Forge** is a deterministic generator for [Pandera](https://pandera.readthedocs.io/) DataFrameModels from pandas DataFrames. It automatically creates exhaustive, type-safe schema definitions without relying on manual work or LLMs, providing a reliable gauge of your dataset's characteristics including statistics, nullability, uniqueness, and patterns.

## Rationale

I have found that when working with LLM's they often fail when working with python code generation for generic dataframes. Especially with feature engineering tasks. With providing an exhaustive schema definition of the dataframe, it helps to ground the LLM and prevents trial and error mistakes when performing analytical tasks.

## Features

- **Automatic Schema Generation**: Convert any pandas DataFrame into a Pandera DataFrameModel
- **Comprehensive Field Analysis**: Detects nullability, uniqueness, min/max values, and data patterns
- **Pattern Detection**: Identifies common patterns in string columns (emails, URLs, phone numbers, etc.)
- **Type Safety**: Generates properly typed fields with appropriate constraints
- **Column Name Sanitization**: Handles problematic column names (spaces, special characters, keywords)
- **Validation**: Validates generated models against source data
- **Extensible**: Optional LLM enrichment for enhanced pattern detection

## Installation

```bash
pip install pandera-forge
```

For LLM enrichment features:
```bash
pip install pandera-forge[llm]
```

## Quick Start

```python
from pandas import DataFrame, to_datetime
from pandera_forge import ModelGenerator

# Create a sample DataFrame
df = DataFrame({
    "customer_id": [1, 2, 3, 4],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com"],
    "age": [25, 30, 35, 40],
    "is_active": [True, True, False, True],
    "signup_date": to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
})

# Generate the model
generator = ModelGenerator()
model_code = generator.generate(df, model_name="CustomerModel")

print(model_code)
```

This generates:

```python
from pandera.pandas import DataFrameModel, Field, Timestamp
from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional


class CustomerModel(DataFrameModel):
    customer_id: Series[Int64] = Field(ge=1, le=4, unique=True, isin=[1,2,3,4])  # 4 distinct values, examples: [1, 2, 3]
    email: Series[Object] = Field(unique=True)  # 4 distinct values, examples: ["alice@example.com", "bob@example.com", "charlie@example.com"], pattern: email
    age: Series[Int64] = Field(ge=25, le=40, unique=True, isin=[25, 30, 35, 40])  # 4 distinct values, examples: [25, 30, 35]
    is_active: Series[Bool] = Field(isin=[True, False])  # 2 distinct values, examples: ["True", "False"]
    signup_date: Series[DateTime] = Field(unique=True)  # 4 distinct values, examples: ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-03 00:00:00"]
```

n.b. the values for the datetimes and emails would also be included in the isin= list, but are omitted for brevity. If the distinct count of a column exceeds 10, the `isin` constraint is omitted.

## Advanced Usage

### Pattern Detection

Pandera Forge automatically detects common patterns in string columns:

```python
from pandas import DataFrame
from pandera_forge import ModelGenerator

df = DataFrame({
    "email": ["user@example.com", "admin@test.org"],
    "phone": ["+1234567890", "+0987654321"],
    "url": ["https://example.com", "https://test.org"]
})

generator = ModelGenerator()
model_code = generator.generate(df, detect_patterns=True)
```

### Handling Messy Data

The generator handles problematic column names and mixed data types:

```python
from pandas import DataFrame
from pandera_forge import ModelGenerator

df = DataFrame({
    "Column With Spaces": [1, 2, 3],
    "123_numeric_start": ["a", "b", "c"],
    "class": [True, False, True],  # Reserved keyword
    "!@#$%": [1.0, 2.0, 3.0]  # Special characters
})

generator = ModelGenerator()
model_code = generator.generate(df)  # Automatically sanitizes column names
```

### LLM Enrichment (Optional)

For enhanced pattern detection and documentation using OpenAI, Anthropic, or local LLMs via Ollama:

```python
from pandas import DataFrame
from pandera_forge import ModelGenerator
df: DataFrame = ... # your DataFrame here

# Using OpenAI (default)
generator = ModelGenerator(llm_api_key="your-openai-api-key")
model_code = generator.generate(df, model_name="EnrichedModel")

# Using Anthropic
from pandera_forge.llm_enricher import LLMEnricher
enricher = LLMEnricher(provider="anthropic", api_key="your-anthropic-api-key")
generator = ModelGenerator(llm_enricher=enricher)
model_code = generator.generate(df, model_name="EnrichedModel")

# Using Ollama (local LLMs - no API key needed)
enricher = LLMEnricher(provider="ollama", model="llama3.2")
generator = ModelGenerator(llm_enricher=enricher)
model_code = generator.generate(df, model_name="EnrichedModel")
```

**Ollama Setup:**
1. Install Ollama: https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2`

## API Reference

### ModelGenerator

Main class for generating Pandera models.

```python
ModelGenerator(
    llm_api_key: Optional[str] = None,
    llm_enricher: Optional[LLMEnricher] = None
)
```

#### Parameters:
- `llm_api_key`: Optional API key for LLM enrichment features (OpenAI by default)
- `llm_enricher`: Optional pre-configured LLMEnricher instance for custom LLM providers

#### Methods:

**generate()**
```python
generate(
    df: DataFrame,
    model_name: str = "DataFrameModel",
    validate: bool = True,
    include_examples: bool = True,
    detect_patterns: bool = True,
    source_file: Optional[Path] = None
) -> Optional[str]
```

Generates a Pandera DataFrameModel from a pandas DataFrame.

**Parameters:**
- `df`: Source DataFrame to generate model from
- `model_name`: Name for the generated model class
- `validate`: Whether to validate the generated model against the source data
- `include_examples`: Whether to include example values in comments
- `detect_patterns`: Whether to detect patterns in string columns
- `source_file`: Optional path to source file for implementation example

**Returns:**
- Generated model code as string, or None if generation failed

### PatternDetector

Detects patterns in string columns.

```python
from pandas import Series

PatternDetector.detect_pattern(
    series: Series, 
    min_match_ratio: float = 0.9
)
```

Supported patterns:
- Email addresses
- URLs
- Phone numbers (US)
- UUIDs
- IPv4 addresses
- Dates (ISO format)
- Credit card numbers
- Hex colors
- MAC addresses
- And more...

## Use Cases

1. **Data Contract Generation**: Automatically generate data contracts from existing datasets
2. **Data Quality Monitoring**: Create schemas for validation in data pipelines
3. **Documentation**: Generate schema documentation for data teams
4. **Testing**: Create test fixtures with proper type constraints
5. **Migration**: Convert existing datasets to validated schemas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built on top of the excellent [Pandera](https://pandera.readthedocs.io/) library for pandas validation.