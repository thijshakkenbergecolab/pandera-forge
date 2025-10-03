# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pandere Forge is a deterministic generator for Pandera DataFrameModels from pandas DataFrames. It automatically creates exhaustive, type-safe schema definitions with statistics, nullability, uniqueness, and pattern detection. The package supports optional LLM enrichment via OpenAI, Anthropic, or local models through Ollama.

## Key Commands

### Development
```bash
# Install package in development mode
pip install -e .

# Install with optional LLM features
pip install -e ".[llm]"

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=pandere_forge --cov-report=term-missing

# Run specific test file
pytest tests/test_generator.py

# Run specific test
pytest tests/test_generator.py::TestModelGenerator::test_basic_generation

# Format code
black pandere_forge tests

# Sort imports
isort pandere_forge tests

# Type checking
mypy pandere_forge

# Linting
flake8 pandere_forge tests
```

## Architecture

### Core Components

The package follows a modular architecture with specialized components:

1. **ModelGenerator** (`generator.py`): Main orchestrator that coordinates all components to generate Pandera models from DataFrames. Handles the complete generation pipeline including validation.

2. **TypeMapper** (`type_mapper.py`): Maps pandas dtypes to Pandera type annotations. Handles special cases like nullable integers and categorical types.

3. **NameSanitizer** (`name_sanitizer.py`): Handles problematic column names by:
   - Converting spaces and special characters to underscores
   - Handling numeric prefixes
   - Avoiding Python reserved keywords
   - Ensuring valid Python identifiers

4. **FieldAnalyzer** (`field_analyzer.py`): Analyzes DataFrame columns to extract:
   - Nullability and uniqueness constraints
   - Min/max values for numeric types
   - Distinct value counts
   - Example values for documentation

5. **PatternDetector** (`pattern_detector.py`): Detects common patterns in string columns using regex:
   - Email, URL, phone numbers, UUIDs
   - IP addresses, dates, credit cards
   - Custom string constraints (min/max length)

6. **CodeGenerator** (`code_generator.py`): Generates Python code strings:
   - Import statements
   - Field definitions with constraints
   - Class definitions
   - Documentation comments

7. **ModelValidator** (`validator.py`): Validates generated models:
   - Syntax validation using AST
   - Runtime validation against source DataFrame

8. **LLMEnricher** (`llm_enricher.py`): Optional component for enhanced pattern detection using LLMs:
   - **OpenAI**: GPT models for cloud-based analysis
   - **Anthropic**: Claude models for cloud-based analysis
   - **Ollama**: Local LLM support for privacy-conscious deployments (llama, mistral, etc.)

### Data Flow

1. DataFrame → FieldAnalyzer extracts column properties
2. Column names → NameSanitizer ensures valid Python identifiers
3. Dtypes → TypeMapper converts to Pandera types
4. String columns → PatternDetector identifies patterns
5. All metadata → CodeGenerator creates Python code
6. Generated code → ModelValidator ensures correctness

### Key Design Decisions

- **Deterministic**: Core functionality doesn't rely on LLMs, ensuring reproducible results
- **Modular**: Each component has a single responsibility, making the code testable and maintainable
- **Defensive**: Handles edge cases like special characters, reserved keywords, and mixed data types
- **Extensible**: LLM enrichment is optional and pluggable
- **Privacy-First**: Supports local LLMs via Ollama for sensitive data that cannot leave the organization

## Testing Strategy

Tests are organized by component with comprehensive coverage:
- `test_generator.py`: Integration tests for complete generation pipeline
- `test_field_analyzer.py`: Column analysis logic
- `test_name_sanitizer.py`: Edge cases for problematic names
- `test_pattern_detector.py`: Pattern detection accuracy
- `test_llm_enricher.py`: LLM integration tests (when implementing)

Each test module uses fixtures for consistent test data and covers both happy paths and edge cases.

## LLM Configuration

The package supports three LLM providers:

### OpenAI (Cloud)
```python
enricher = LLMEnricher(provider="openai", api_key="sk-...")
```

### Anthropic (Cloud)
```python
enricher = LLMEnricher(provider="anthropic", api_key="sk-ant-...")
```

### Ollama (Local)
```python
# No API key needed - connects to local Ollama server
enricher = LLMEnricher(provider="ollama", model="llama3.2")
```

For Ollama setup:
1. Install from https://ollama.ai
2. Run `ollama serve`
3. Pull models: `ollama pull llama3.2`