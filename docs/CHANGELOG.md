# Changelog

All notable changes to Pandera Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Databricks Support**: Full integration with Databricks including Unity Catalog support
  - New `DatabricksGenerator` class for generating models from Databricks tables
  - `DatabricksConnector` for handling authentication and connections
  - Support for Delta Lake table paths with version support
  - Bulk model generation for entire catalogs/schemas
  - Authentication via direct credentials, environment variables, or CLI profiles
- **Apache Spark Support**: Native support for Spark DataFrames
  - New `SparkGenerator` for PySpark DataFrames
  - `SparkTypeMapper` for Spark SQL type mappings
  - `SparkFieldAnalyzer` for analyzing Spark DataFrame columns
  - Automatic sampling for pattern detection in large datasets
- **Modular Architecture**: Complete refactoring to support multiple DataFrame backends
  - New `BaseGenerator` abstract class for extensibility
  - `BaseFieldAnalyzer` abstract class for backend-specific column analysis
  - `BaseTypeMapper` abstract class for backend-specific type mappings
  - Main `ModelGenerator` now acts as a factory that delegates to appropriate backend
- **Factory Methods**: Convenient class methods for creating backend-specific generators
  - `ModelGenerator.create_for_spark()` for Spark DataFrames
  - `ModelGenerator.create_for_databricks()` for Databricks integration
- **Comprehensive Documentation**: Added detailed Databricks integration guide
  - README_DATABRICKS.md with usage examples and best practices
  - Examples for various Databricks use cases (single table, multiple tables, Delta Lake)
  - Performance considerations and troubleshooting guide
- **Optional Dependencies**: New extras for Spark and Databricks support
  - `pip install pandera-forge[spark]` for Spark support
  - `pip install pandera-forge[databricks]` for full Databricks integration

### Changed
- **Breaking**: Main `ModelGenerator` constructor now accepts `backend` parameter
- **Breaking**: Internal components moved to backend-specific modules
  - `PandasGenerator` in `pandera_forge.pandas`
  - `SparkGenerator` in `pandera_forge.spark`
  - Core abstractions in `pandera_forge.core`
- **Test Structure**: Reorganized tests into backend-specific directories
  - `tests/core/` for core functionality
  - `tests/pandas/` for pandas-specific tests
  - `tests/spark/` for Spark-specific tests
  - `tests/databricks/` for Databricks integration tests
- **Dependencies**: Updated pyproject.toml with spark and databricks extras

### Improved
- **Backward Compatibility**: Main `ModelGenerator` API remains unchanged for pandas usage
- **Performance**: Automatic caching and sampling for large Spark DataFrames
- **Type Safety**: Better handling of complex Spark types (arrays, maps, structs)

## [0.1.0] - 2025-01-30

### Added
- Initial release with pandas DataFrame support
- Automatic Pandera DataFrameModel generation
- Type detection and mapping
- Nullability and uniqueness detection
- Pattern detection for string columns (emails, URLs, UUIDs, etc.)
- Min/max constraints for numeric types
- Optional LLM enrichment via OpenAI, Anthropic, or Ollama
- Excel file support with multi-sheet handling
- CSV file support with encoding detection
- Model validation against source DataFrames
- Comprehensive test suite

### Features
- Deterministic schema generation without LLMs
- Column name sanitization (handles special characters, reserved keywords)
- Example value extraction for documentation
- AST-based syntax validation
- Runtime validation against source data
