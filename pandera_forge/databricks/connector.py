"""
Databricks connection and authentication utilities
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path


class DatabricksConnector:
    """Handles Databricks authentication and connection setup"""

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        cluster_id: Optional[str] = None,
        sql_endpoint_id: Optional[str] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ):
        """
        Initialize Databricks connector.

        Args:
            host: Databricks workspace URL (e.g., https://xxx.cloud.databricks.com)
            token: Databricks personal access token
            cluster_id: ID of the compute cluster to use
            sql_endpoint_id: ID of the SQL warehouse endpoint
            catalog: Unity Catalog name (for Unity Catalog enabled workspaces)
            schema: Schema/database name
        """
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.cluster_id = cluster_id or os.getenv("DATABRICKS_CLUSTER_ID")
        self.sql_endpoint_id = sql_endpoint_id or os.getenv("DATABRICKS_SQL_ENDPOINT_ID")
        self.catalog = catalog or os.getenv("DATABRICKS_CATALOG", "hive_metastore")
        self.schema = schema or os.getenv("DATABRICKS_SCHEMA", "default")

        self._spark_session = None
        self._sql_connection = None

    def get_spark_session(self, app_name: str = "PanderaForge"):
        """
        Get or create a Spark session configured for Databricks.

        Args:
            app_name: Name for the Spark application

        Returns:
            Configured SparkSession
        """
        if self._spark_session is not None:
            return self._spark_session

        try:
            from pyspark.sql import SparkSession

            # Check if we're running in Databricks
            if self._is_databricks_runtime():
                # Use the existing Databricks Spark session
                self._spark_session = SparkSession.builder.getOrCreate()
            else:
                # Create a remote Spark session connecting to Databricks
                builder = SparkSession.builder.appName(app_name)

                if self.host and self.token:
                    # Configure Databricks Connect
                    builder = builder \
                        .config("spark.databricks.service.address", self.host) \
                        .config("spark.databricks.service.token", self.token)

                    if self.cluster_id:
                        builder = builder.config("spark.databricks.service.clusterId", self.cluster_id)

                    # Set catalog and schema if provided
                    if self.catalog:
                        builder = builder.config("spark.sql.catalog.spark_catalog", self.catalog)
                    if self.schema:
                        builder = builder.config("spark.sql.defaultDatabase", self.schema)

                self._spark_session = builder.getOrCreate()

            # Set the default catalog and schema
            if self.catalog and self.schema:
                self._spark_session.sql(f"USE CATALOG {self.catalog}")
                self._spark_session.sql(f"USE SCHEMA {self.schema}")

            return self._spark_session

        except ImportError:
            raise ImportError(
                "PySpark not installed. Install with: pip install pyspark "
                "or pip install databricks-connect"
            )

    def get_sql_connection(self):
        """
        Get a SQL connection to Databricks SQL warehouse.

        Returns:
            Databricks SQL connection object
        """
        if self._sql_connection is not None:
            return self._sql_connection

        try:
            from databricks import sql

            if not self.host or not self.token:
                raise ValueError("Host and token required for SQL connection")

            if not self.sql_endpoint_id:
                raise ValueError("SQL endpoint ID required for SQL connection")

            self._sql_connection = sql.connect(
                server_hostname=self.host.replace("https://", "").replace("http://", ""),
                http_path=f"/sql/1.0/endpoints/{self.sql_endpoint_id}",
                access_token=self.token,
                catalog=self.catalog,
                schema=self.schema,
            )

            return self._sql_connection

        except ImportError:
            raise ImportError(
                "databricks-sql-connector not installed. "
                "Install with: pip install databricks-sql-connector"
            )

    def list_tables(self, catalog: Optional[str] = None, schema: Optional[str] = None) -> list:
        """
        List tables in the specified catalog and schema.

        Args:
            catalog: Catalog name (uses default if not provided)
            schema: Schema name (uses default if not provided)

        Returns:
            List of table names
        """
        spark = self.get_spark_session()
        catalog = catalog or self.catalog
        schema = schema or self.schema

        try:
            # Use Spark SQL to list tables
            if catalog and catalog != "hive_metastore":
                tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{schema}")
            else:
                tables_df = spark.sql(f"SHOW TABLES IN {schema}")

            return [row.tableName for row in tables_df.collect()]
        except Exception as e:
            print(f"Error listing tables: {e}")
            return []

    def read_table(
        self,
        table_name: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        sample_fraction: Optional[float] = None
    ):
        """
        Read a table from Databricks.

        Args:
            table_name: Name of the table
            catalog: Catalog name (uses default if not provided)
            schema: Schema name (uses default if not provided)
            sample_fraction: Optional fraction to sample (0.0 to 1.0)

        Returns:
            Spark DataFrame
        """
        spark = self.get_spark_session()
        catalog = catalog or self.catalog
        schema = schema or self.schema

        # Build the full table name
        if catalog and catalog != "hive_metastore":
            full_table_name = f"{catalog}.{schema}.{table_name}"
        else:
            full_table_name = f"{schema}.{table_name}"

        df = spark.table(full_table_name)

        # Sample if requested
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            df = df.sample(withReplacement=False, fraction=sample_fraction)

        return df

    def read_delta_table(self, path: str, version: Optional[int] = None):
        """
        Read a Delta table from a path.

        Args:
            path: Path to the Delta table
            version: Optional version number to read

        Returns:
            Spark DataFrame
        """
        spark = self.get_spark_session()

        if version is not None:
            return spark.read.format("delta").option("versionAsOf", version).load(path)
        else:
            return spark.read.format("delta").load(path)

    def close(self):
        """Close connections"""
        if self._sql_connection:
            self._sql_connection.close()
            self._sql_connection = None

        if self._spark_session:
            self._spark_session.stop()
            self._spark_session = None

    def _is_databricks_runtime(self) -> bool:
        """Check if running in Databricks Runtime"""
        return "DATABRICKS_RUNTIME_VERSION" in os.environ

    @classmethod
    def from_profile(cls, profile_name: str = "DEFAULT"):
        """
        Create a connector from a Databricks CLI profile.

        Args:
            profile_name: Name of the profile in ~/.databrickscfg

        Returns:
            DatabricksConnector instance
        """
        config = cls._read_databricks_config(profile_name)
        return cls(
            host=config.get("host"),
            token=config.get("token"),
            cluster_id=config.get("cluster_id"),
            sql_endpoint_id=config.get("sql_endpoint_id"),
        )

    @staticmethod
    def _read_databricks_config(profile_name: str = "DEFAULT") -> Dict[str, str]:
        """
        Read Databricks configuration from ~/.databrickscfg

        Args:
            profile_name: Profile name to read

        Returns:
            Dictionary with configuration values
        """
        config_path = Path.home() / ".databrickscfg"
        config = {}

        if not config_path.exists():
            return config

        current_profile = None
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_profile = line[1:-1]
                elif current_profile == profile_name and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()

        return config

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()