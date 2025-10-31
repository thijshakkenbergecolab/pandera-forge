"""
Tests for Spark-specific components
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pandera_forge.spark.spark_type_mapper import SparkTypeMapper
from pandera_forge.spark.spark_field_analyzer import SparkFieldAnalyzer
from pandera_forge.spark.spark_generator import SparkGenerator


class TestSparkTypeMapper:
    """Test Spark type mapping functionality"""

    def test_spark_to_pandera_mapping(self):
        """Test mapping of Spark SQL types to Pandera types"""
        mapper = SparkTypeMapper()

        # Test basic type mappings
        assert mapper.get_pandera_type("int").__name__ == "Int32"
        assert mapper.get_pandera_type("bigint").__name__ == "Int64"
        assert mapper.get_pandera_type("string").__name__ == "String"
        assert mapper.get_pandera_type("boolean").__name__ == "Bool"
        assert mapper.get_pandera_type("double").__name__ == "Float64"
        assert mapper.get_pandera_type("float").__name__ == "Float32"
        assert mapper.get_pandera_type("timestamp").__name__ == "Timestamp"

    def test_normalize_dtype(self):
        """Test normalization of Spark data types"""
        mapper = SparkTypeMapper()

        # Test string normalization
        assert mapper.normalize_dtype("INT") == "int"
        assert mapper.normalize_dtype("STRING") == "string"

        # Test parameterized type normalization
        assert mapper.normalize_dtype("decimal(10,2)") == "decimal"
        assert mapper.normalize_dtype("varchar(255)") == "varchar"

        # Test array type normalization
        assert mapper.normalize_dtype("array<string>") == "array"
        assert mapper.normalize_dtype("map<string,int>") == "map"

    def test_complex_types(self):
        """Test mapping of complex Spark types"""
        mapper = SparkTypeMapper()

        # Complex types should map to Object
        assert mapper.get_pandera_type("array").__name__ == "Object"
        assert mapper.get_pandera_type("map").__name__ == "Object"
        assert mapper.get_pandera_type("struct").__name__ == "Object"
        assert mapper.get_pandera_type("binary").__name__ == "Object"


class TestSparkFieldAnalyzer:
    """Test Spark field analysis functionality"""

    def test_analyze_column_numeric(self):
        """Test analyzing a numeric Spark column"""
        try:
            import pyspark.sql.functions as F
        except ImportError:
            pytest.skip("PySpark not available")

        with patch('pyspark.sql.functions') as mock_f:
            # Create mock DataFrame and column
            mock_df = Mock()
            mock_df.count.return_value = 100

            # Mock schema
            mock_field = Mock()
            mock_field.name = "test_col"
            mock_field.dataType.simpleString.return_value = "int"
            mock_df.schema.fields = [mock_field]

            # Mock filter and distinct for nullability and uniqueness checks
            mock_df.filter.return_value.count.return_value = 0  # No nulls
            mock_df.select.return_value.distinct.return_value.count.return_value = 100

            # Mock stats collection row
            mock_stats_row = MagicMock()
            mock_stats_row.__getitem__ = Mock(side_effect=lambda x: {"min": 1, "max": 100}[x])

            # Create a separate mock for select that returns stats
            mock_select_for_stats = Mock()
            mock_select_for_stats.collect.return_value = [mock_stats_row]

            # Create a counter to alternate between distinct count select and stats select
            select_call_count = [0]
            def select_side_effect(*args, **kwargs):
                select_call_count[0] += 1
                if select_call_count[0] == 1:
                    # First call is for distinct count
                    return mock_df.select.return_value
                else:
                    # Second call is for stats collection
                    return mock_select_for_stats

            mock_df.select.side_effect = select_side_effect

            # Mock F.col and other functions
            mock_f.col.return_value.isNull.return_value = Mock()
            mock_f.min.return_value.alias.return_value = Mock()
            mock_f.max.return_value.alias.return_value = Mock()

            analyzer = SparkFieldAnalyzer()
            properties = analyzer.analyze_column(mock_df, "test_col")

            assert properties["is_nullable"] == False
            assert properties["is_unique"] == True
            assert properties["distinct_count"] == 100
            assert properties["min_value"] == 1
            assert properties["max_value"] == 100

    def test_get_column_dtype(self):
        """Test getting column data type from Spark DataFrame"""
        # Create mock DataFrame with schema
        mock_df = Mock()
        mock_field = Mock()
        mock_field.name = "test_col"
        mock_field.dataType.simpleString.return_value = "string"
        mock_df.schema.fields = [mock_field]

        analyzer = SparkFieldAnalyzer()
        dtype = analyzer.get_column_dtype(mock_df, "test_col")

        assert dtype == "string"

    def test_is_numeric_column(self):
        """Test checking if a Spark column is numeric"""
        # Create mock DataFrame
        mock_df = Mock()

        # Mock numeric column
        mock_field_numeric = Mock()
        mock_field_numeric.name = "numeric_col"
        mock_field_numeric.dataType.simpleString.return_value = "double"

        # Mock string column
        mock_field_string = Mock()
        mock_field_string.name = "string_col"
        mock_field_string.dataType.simpleString.return_value = "string"

        analyzer = SparkFieldAnalyzer()

        # Test numeric column
        mock_df.schema.fields = [mock_field_numeric]
        assert analyzer.is_numeric_column(mock_df, "numeric_col") == True

        # Test string column
        mock_df.schema.fields = [mock_field_string]
        assert analyzer.is_numeric_column(mock_df, "string_col") == False


class TestSparkGenerator:
    """Test Spark generator functionality"""

    def test_generate_with_spark_dataframe(self):
        """Test generating model from Spark DataFrame"""
        try:
            from pyspark.sql import DataFrame as SparkDataFrame

            # Create mock Spark DataFrame with spec
            mock_df = Mock(spec=SparkDataFrame)
            mock_df.columns = ["col1", "col2"]
            mock_df.cache.return_value = mock_df
            mock_df.unpersist.return_value = mock_df

            # Mock schema for type detection
            mock_field1 = Mock()
            mock_field1.name = "col1"
            mock_field1.dataType.simpleString.return_value = "int"
            mock_field2 = Mock()
            mock_field2.name = "col2"
            mock_field2.dataType.simpleString.return_value = "string"
            mock_df.schema.fields = [mock_field1, mock_field2]

            generator = SparkGenerator()

            # Mock the field generation
            with patch.object(generator, '_generate_field', return_value="col1: Int64 = Field()"):
                result = generator.generate(mock_df, model_name="TestModel", validate=False)

                assert result is not None
                assert "TestModel" in result
                assert "Generated from Spark DataFrame" in result or "TestModel" in result
        except ImportError:
            # Skip if PySpark not available
            pytest.skip("PySpark not available")

    def test_get_columns(self):
        """Test getting column names from Spark DataFrame"""
        mock_df = Mock()
        mock_df.columns = ["col1", "col2", "col3"]

        generator = SparkGenerator()
        columns = generator.get_columns(mock_df)

        assert columns == ["col1", "col2", "col3"]

    def test_from_table(self):
        """Test generating model from Spark table"""
        # Create mock SparkSession
        mock_session = Mock()
        mock_table_df = Mock()
        mock_table_df.columns = ["id", "name"]
        mock_session.table.return_value = mock_table_df

        generator = SparkGenerator()

        with patch.object(generator, 'generate', return_value="model_code") as mock_generate:
            result = generator.from_table("test_table", mock_session)

            mock_session.table.assert_called_once_with("test_table")
            mock_generate.assert_called_once()
            assert result == "model_code"

    def test_from_parquet(self):
        """Test generating model from Parquet file"""
        # Create mock SparkSession
        mock_session = Mock()
        mock_parquet_df = Mock()
        mock_parquet_df.columns = ["col1", "col2"]
        mock_session.read.parquet.return_value = mock_parquet_df

        generator = SparkGenerator()

        with patch.object(generator, 'generate', return_value="model_code") as mock_generate:
            result = generator.from_parquet("/path/to/file.parquet", mock_session)

            mock_session.read.parquet.assert_called_once_with("/path/to/file.parquet")
            mock_generate.assert_called_once()
            assert result == "model_code"


class TestDatabricksConnector:
    """Test Databricks connector functionality"""

    def test_get_spark_session(self):
        """Test creating Spark session for Databricks"""
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            pytest.skip("PySpark not available")

        from pandera_forge.databricks.connector import DatabricksConnector

        with patch('pyspark.sql.SparkSession') as mock_spark_session_class:
            # Mock SparkSession builder
            mock_builder = Mock()
            mock_session = Mock()
            mock_sql_result = Mock()
            mock_session.sql.return_value = mock_sql_result
            mock_builder.getOrCreate.return_value = mock_session
            mock_builder.appName.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_spark_session_class.builder = mock_builder

            connector = DatabricksConnector(
                host="https://test.cloud.databricks.com",
                token="test-token",
                cluster_id="test-cluster"
            )

            # Test in non-Databricks environment
            with patch.object(connector, '_is_databricks_runtime', return_value=False):
                session = connector.get_spark_session()
                assert session == mock_session

    def test_is_databricks_runtime(self):
        """Test detecting Databricks runtime environment"""
        from pandera_forge.databricks.connector import DatabricksConnector

        connector = DatabricksConnector()

        # Test without Databricks runtime
        with patch.dict('os.environ', {}, clear=True):
            assert connector._is_databricks_runtime() == False

        # Test with Databricks runtime
        with patch.dict('os.environ', {'DATABRICKS_RUNTIME_VERSION': '14.0'}, clear=True):
            assert connector._is_databricks_runtime() == True

    def test_read_databricks_config(self):
        """Test reading Databricks configuration file"""
        from pandera_forge.databricks.connector import DatabricksConnector
        from io import StringIO

        mock_config = """[DEFAULT]
host = https://test.cloud.databricks.com
token = test-token

[PROFILE1]
host = https://other.cloud.databricks.com
token = other-token
"""

        mock_open = Mock(return_value=StringIO(mock_config))
        with patch('builtins.open', mock_open):
            with patch('pathlib.Path.exists', return_value=True):
                config = DatabricksConnector._read_databricks_config("DEFAULT")
                assert config["host"] == "https://test.cloud.databricks.com"
                assert config["token"] == "test-token"

        # Reset for second test
        mock_open = Mock(return_value=StringIO(mock_config))
        with patch('builtins.open', mock_open):
            with patch('pathlib.Path.exists', return_value=True):
                config = DatabricksConnector._read_databricks_config("PROFILE1")
                assert config["host"] == "https://other.cloud.databricks.com"
                assert config["token"] == "other-token"