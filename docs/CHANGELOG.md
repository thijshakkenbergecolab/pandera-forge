# Changelog

All notable changes to Pandera Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-10-31

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

### Fixed
- Fixed Spark and Databricks test mocking issues (952f2f6)
  - Fixed import paths for pyspark.sql.functions patching
  - Fixed Mock spec for SparkDataFrame to work with isinstance checks
  - Fixed Databricks config file parsing mock using StringIO
  - All 103 tests now pass with 62% overall coverage

## [0.1.5] - 2025-10-06

### Changed
- Updated documentation examples (ca74803)
- Installed all dependencies in development environment (db87f91)

### Fixed
- Version bumps and package configuration (626c67c, 6e4c9cf)

## [0.1.4] - 2025-10-05

### Added
- Excel file support with dependencies (cb84b80, f423c56)
  - Multi-sheet Excel file handling
  - XLRDError handling for unsupported formats
  - Support for .xlsx and .xls formats
- CI/CD pipeline setup (cd78616)
- Extended test coverage (34e5eeb)
- PyPI package metadata improvements (bd9dc2b, d3b94f4)

### Changed
- Support for Python 3.9+ only (3eca839)
- Better import organization (538a2fd)
- Improved type hints and linting (ad3fffb)

### Fixed
- Test fixes and improvements (328801d, 0f68281, 992ad6a, 009e0cc)
- Import fixes for better module structure (6406803)

## [0.1.3] - 2025-10-05

### Added
- Rationale section in README explaining use case with LLMs (ce1ffba)
- Example values moved to isin constraint for better validation (6df6d1d)

### Improved
- README code examples and documentation (a0149e1)

### Fixed
- Nullable field test fixes (f40cb5c)

## [0.1.2] - 2025-10-05

### Changed
- Moved import line to constants for consistency (e5d09d6, f59dd87)

### Improved
- Increased test coverage (0eb0659)

## [0.1.1] - 2025-10-05

### Fixed
- Import fixes and module organization (6406803)
- Example file renaming (9fb0fee)

## [0.1.0] - 2025-10-04

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

### Changed
- Package renamed from pandere-forge to pandera-forge (dbeed44)

## [0.0.1] - 2025-10-04

### Added
- Initial commit with core functionality (0793f3f)
- Basic Pandera DataFrameModel generator
- Ollama support for LLM enrichment
- Type mapper for pandas dtype conversion
- Field analyzer for column properties
- Pattern detector for string patterns
- Name sanitizer for column names
- Code generator for model creation
- Model validator for schema validation
