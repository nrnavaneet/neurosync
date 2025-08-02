"""
Database connector for SQL databases and data warehouses
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Data warehouse specific imports
try:
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

from neurosync.core.exceptions.custom_exceptions import ConnectionError, IngestionError
from neurosync.ingestion.base.connector import (
    BaseConnector,
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceType,
)


class DatabaseConnector(BaseConnector):
    """Connector for SQL databases and modern data warehouses"""

    # Supported database and data warehouse types
    SUPPORTED_DATABASES = {
        "postgresql",
        "mysql",
        "sqlite",
        "oracle",
        "mssql",
        "snowflake",
        "bigquery",
        "redshift",
        "databricks",
        "clickhouse",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_string = config.get("connection_string", "")
        self.database_type = config.get("database_type", "").lower()
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database", "")
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.schema = config.get("schema", "public")
        self.tables = config.get("tables", [])
        self.queries = config.get("queries", [])
        self.batch_size = config.get("batch_size", 1000)
        self.max_rows = config.get("max_rows", 100000)
        self.connection_pool_size = config.get("connection_pool_size", 5)

        # Data warehouse specific configurations
        self.warehouse = config.get("warehouse", "")  # Snowflake
        self.role = config.get("role", "")  # Snowflake
        self.account = config.get("account", "")  # Snowflake
        self.project_id = config.get("project_id", "")  # BigQuery
        self.credentials_path = config.get("credentials_path", "")  # BigQuery
        self.cluster_id = config.get("cluster_id", "")  # Databricks
        self.http_path = config.get("http_path", "")  # Databricks
        self.token = config.get("token", "")  # Databricks/PAT tokens

        self.engine = None
        self.session_maker = None
        self.bigquery_client = None  # For BigQuery
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate database connector configuration"""
        if not self.connection_string and not self.database_type:
            raise IngestionError(
                "Either connection_string or database_type is required"
            )

        if self.database_type and self.database_type not in self.SUPPORTED_DATABASES:
            raise IngestionError(
                f"Unsupported database type: {self.database_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_DATABASES)}"
            )

        if not self.connection_string:
            # Data warehouse specific validations
            if self.database_type == "snowflake":
                required_fields = [
                    "account",
                    "username",
                    "password",
                    "database",
                    "warehouse",
                ]
                for field in required_fields:
                    if not getattr(self, field):
                        raise IngestionError(f"{field} is required for Snowflake")
            elif self.database_type == "bigquery":
                if not self.project_id:
                    raise IngestionError("project_id is required for BigQuery")
            elif self.database_type == "databricks":
                required_fields = ["host", "http_path", "token"]
                for field in required_fields:
                    if not getattr(self, field):
                        raise IngestionError(f"{field} is required for Databricks")
            elif self.database_type == "redshift":
                required_fields = ["host", "database", "username", "password"]
                for field in required_fields:
                    if not getattr(self, field):
                        raise IngestionError(f"{field} is required for Redshift")
            elif self.database_type == "sqlite":
                # SQLite only requires database path
                if not self.database:
                    raise IngestionError("database path is required for SQLite")
            else:
                # Traditional databases (PostgreSQL, MySQL, etc.)
                required_fields = ["host", "database", "username"]
                for field in required_fields:
                    if not getattr(self, field):
                        raise IngestionError(
                            f"{field} is required when not using connection_string"
                        )

    def _build_connection_string(self) -> str:
        """Build connection string from individual components"""
        if self.connection_string:
            return self.connection_string

        # Traditional SQL databases
        if self.database_type == "postgresql":
            return (
                f"postgresql+psycopg://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        elif self.database_type == "mysql":
            return (
                f"mysql+aiomysql://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        elif self.database_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.database}"
        elif self.database_type == "oracle":
            return (
                f"oracle+cx_oracle://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        elif self.database_type == "mssql":
            driver = "ODBC+Driver+17+for+SQL+Server"
            return (
                f"mssql+pyodbc://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}?driver={driver}"
            )

        # Data warehouses
        elif self.database_type == "snowflake":
            if SNOWFLAKE_AVAILABLE:
                return (
                    f"snowflake://{self.username}:{self.password}@{self.account}"
                    f"/{self.database}/{self.schema}?warehouse={self.warehouse}"
                    f"&role={self.role}"
                )
            else:
                raise IngestionError(
                    "Snowflake connector not available. "
                    "Install: pip install snowflake-sqlalchemy"
                )

        elif self.database_type == "redshift":
            port = self.port or 5439
            return (
                f"redshift+psycopg2://{self.username}:{self.password}"
                f"@{self.host}:{port}/{self.database}"
            )

        elif self.database_type == "databricks":
            return (
                f"databricks+connector://token:{self.token}@{self.host}:443"
                f"/{self.database}?http_path={self.http_path}"
            )

        elif self.database_type == "clickhouse":
            port = self.port or 9000
            return (
                f"clickhouse+native://{self.username}:{self.password}"
                f"@{self.host}:{port}/{self.database}"
            )

        elif self.database_type == "bigquery":
            if not BIGQUERY_AVAILABLE:
                raise IngestionError(
                    "BigQuery connector not available. "
                    "Install: pip install google-cloud-bigquery"
                )
            # BigQuery uses a different connection approach
            return f"bigquery://{self.project_id}"

        else:
            raise IngestionError(f"Unsupported database type: {self.database_type}")

    async def connect(self) -> None:
        """Establish database connection"""
        try:
            # Special handling for BigQuery
            if self.database_type == "bigquery":
                if BIGQUERY_AVAILABLE:
                    if self.credentials_path:
                        credentials = (
                            service_account.Credentials.from_service_account_file(
                                self.credentials_path
                            )
                        )
                        self.bigquery_client = bigquery.Client(
                            project=self.project_id, credentials=credentials
                        )
                    else:
                        # Use default credentials
                        self.bigquery_client = bigquery.Client(project=self.project_id)

                    # Test connection
                    if self.bigquery_client:
                        list(self.bigquery_client.list_datasets(max_results=1))
                        self.logger.info(
                            f"Connected to BigQuery project: {self.project_id}"
                        )
                    return
                else:
                    raise IngestionError(
                        "BigQuery connector not available. "
                        "Install: pip install google-cloud-bigquery"
                    )

            # Standard SQLAlchemy connection for other databases
            connection_string = self._build_connection_string()

            # Special pool settings for data warehouses
            if self.database_type in ["snowflake", "redshift", "databricks"]:
                pool_size = min(
                    self.connection_pool_size, 3
                )  # Lower pool size for warehouses
                max_overflow = 0
                pool_timeout = 30
            else:
                pool_size = self.connection_pool_size
                max_overflow = 0
                pool_timeout = 10

            self.engine = create_async_engine(
                connection_string,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                echo=False,
            )

            self.session_maker = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )

            # Test connection
            if self.engine:
                async with self.engine.begin() as conn:
                    test_query = self._get_test_query()
                    await conn.execute(text(test_query))

                self.logger.info(
                    f"Connected to {self.database_type} database: "
                    f"{self.host or self.account}/{self.database}"
                )

        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.database_type}: {e}")

    def _get_test_query(self) -> str:
        """Get appropriate test query for the database type"""
        if self.database_type in ["snowflake", "bigquery"]:
            return "SELECT 1 as test"
        elif self.database_type == "databricks":
            return "SELECT 1 as test"
        elif self.database_type == "oracle":
            return "SELECT 1 FROM DUAL"
        else:
            return "SELECT 1"

    async def disconnect(self) -> None:
        """Close database connection"""
        if self.bigquery_client:
            self.bigquery_client.close()
            self.bigquery_client = None
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_maker = None
        self.logger.info("Disconnected from database")

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if self.database_type == "bigquery":
                if not self.bigquery_client:
                    await self.connect()
                # Test BigQuery connection
                if self.bigquery_client:
                    list(self.bigquery_client.list_datasets(max_results=1))
                    return True
                return False
            else:
                if not self.engine:
                    await self.connect()

                if self.engine:
                    async with self.engine.begin() as conn:
                        test_query = self._get_test_query()
                        await conn.execute(text(test_query))
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def list_sources(self) -> List[str]:
        """List available tables and configured queries"""
        sources = []

        # Add configured tables
        if self.tables:
            sources.extend([f"table:{table}" for table in self.tables])

        # Add configured queries
        if self.queries:
            for query_config in self.queries:
                query_name = query_config.get("name", f"query_{len(sources)}")
                sources.append(f"query:{query_name}")

        # If no specific tables/queries configured, list all tables
        if not sources:
            try:
                table_names = await self._list_all_tables()
                sources.extend([f"table:{table}" for table in table_names])
            except Exception as e:
                self.logger.warning(f"Failed to list tables: {e}")

        return sources

    async def _list_all_tables(self) -> List[str]:
        """List all tables in the database"""
        try:
            # BigQuery special handling
            if self.database_type == "bigquery":
                if not self.bigquery_client:
                    await self.connect()

                if not self.bigquery_client:
                    return []

                tables = []
                datasets = list(self.bigquery_client.list_datasets())

                for dataset in datasets:
                    dataset_tables = list(
                        self.bigquery_client.list_tables(dataset.dataset_id)
                    )
                    for table in dataset_tables:
                        tables.append(f"{dataset.dataset_id}.{table.table_id}")

                return tables

            # Standard SQL databases and warehouses
            if not self.engine:
                await self.connect()

            if not self.engine:
                return []

            async with self.engine.begin() as conn:
                if self.database_type in ["postgresql", "redshift"]:
                    query = text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                    )
                    result = await conn.execute(query, {"schema": self.schema})

                elif self.database_type == "mysql":
                    query = text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :database
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                    )
                    result = await conn.execute(query, {"database": self.database})

                elif self.database_type == "sqlite":
                    query = text(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                    """
                    )
                    result = await conn.execute(query)

                elif self.database_type == "snowflake":
                    query = text(
                        """
                        SHOW TABLES IN SCHEMA IDENTIFIER(:schema)
                    """
                    )
                    try:
                        result = await conn.execute(
                            query, {"schema": f"{self.database}.{self.schema}"}
                        )
                        return [
                            row[1] for row in result.fetchall()
                        ]  # Table name is in second column
                    except Exception:
                        # Fallback to information_schema
                        query = text(
                            """
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = :schema
                            AND table_type = 'BASE TABLE'
                            ORDER BY table_name
                        """
                        )
                        result = await conn.execute(
                            query, {"schema": self.schema.upper()}
                        )

                elif self.database_type == "databricks":
                    query = text(
                        """
                        SHOW TABLES IN :schema
                    """
                    )
                    try:
                        result = await conn.execute(query, {"schema": self.schema})
                        return [
                            row[1] for row in result.fetchall()
                        ]  # Table name is in second column
                    except Exception:
                        # Fallback method
                        query = text(
                            """
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_schema = :schema
                            ORDER BY table_name
                        """
                        )
                        result = await conn.execute(query, {"schema": self.schema})

                elif self.database_type == "oracle":
                    query = text(
                        """
                        SELECT table_name
                        FROM user_tables
                        ORDER BY table_name
                    """
                    )
                    result = await conn.execute(query)

                elif self.database_type == "mssql":
                    query = text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_type = 'BASE TABLE'
                        AND table_schema = :schema
                        ORDER BY table_name
                    """
                    )
                    result = await conn.execute(query, {"schema": self.schema})

                elif self.database_type == "clickhouse":
                    query = text(
                        """
                        SELECT name
                        FROM system.tables
                        WHERE database = :database
                        ORDER BY name
                    """
                    )
                    result = await conn.execute(query, {"database": self.database})

                else:
                    return []

                return [row[0] for row in result.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to list tables: {e}")
            return []

    async def ingest(self, source_id: str, **kwargs) -> IngestionResult:
        """Ingest data from database table or query"""
        start_time = time.time()

        try:
            source_type, source_name = source_id.split(":", 1)

            if source_type == "table":
                result = await self._ingest_table(source_name, **kwargs)
            elif source_type == "query":
                result = await self._ingest_query(source_name, **kwargs)
            else:
                raise IngestionError(f"Unknown source type: {source_type}")

            result.processing_time_seconds = time.time() - start_time
            self.logger.info(f"Successfully ingested database source: {source_id}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to ingest database source {source_id}: {e}")
            return IngestionResult(
                success=False,
                source_id=source_id,
                error=str(e),
                processing_time_seconds=processing_time,
            )

    async def _ingest_table(self, table_name: str, **kwargs) -> IngestionResult:
        """Ingest data from a database table"""
        if not self.engine:
            await self.connect()

        limit = kwargs.get("limit", self.max_rows)
        offset = kwargs.get("offset", 0)
        where_clause = kwargs.get("where", "")
        order_by = kwargs.get("order_by", "")

        # Build query
        query_parts = [f"SELECT * FROM {table_name}"]

        if where_clause:
            query_parts.append(f"WHERE {where_clause}")

        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        if offset:
            query_parts.append(f"OFFSET {offset}")

        query_sql = " ".join(query_parts)

        # Execute query and fetch results
        rows_data: List[Dict[str, Any]] = []
        total_size = 0

        if not self.engine:
            await self.connect()

        if not self.engine:
            raise IngestionError("Failed to establish database connection")

        async with self.engine.begin() as conn:
            result = await conn.execute(text(query_sql))
            columns = result.keys()

            async for row in result:
                row_dict = dict(zip(columns, row))
                rows_data.append(row_dict)
                total_size += len(str(row_dict))

        # Convert to JSON format
        content = json.dumps(rows_data, indent=2, default=str, ensure_ascii=False)

        # Create metadata
        metadata = self._create_source_metadata(
            source_id=f"table:{table_name}",
            source_type=SourceType.DATABASE,
            content_type=ContentType.JSON,
            custom_metadata={
                "table_name": table_name,
                "row_count": len(rows_data),
                "columns": list(columns),
                "query": query_sql,
                "database_type": self.database_type,
                "schema": self.schema,
            },
        )

        return IngestionResult(
            success=True,
            source_id=f"table:{table_name}",
            content=content,
            metadata=metadata,
            raw_size_bytes=total_size,
            processed_size_bytes=len(content.encode("utf-8")),
        )

    async def _ingest_query(self, query_name: str, **kwargs) -> IngestionResult:
        """Ingest data from a configured query"""
        query_config = self._find_query_config(query_name)
        if not query_config:
            raise IngestionError(f"Query configuration not found: {query_name}")

        query_sql = query_config["sql"]
        parameters = query_config.get("parameters", {})
        parameters.update(kwargs.get("parameters", {}))

        if not self.engine:
            await self.connect()

        if not self.engine:
            raise IngestionError("Failed to establish database connection")

        # Execute query
        rows_data = []
        total_size = 0

        async with self.engine.begin() as conn:
            result = await conn.execute(text(query_sql), parameters)
            columns = result.keys()

            row_count = 0
            async for row in result:
                if row_count >= self.max_rows:
                    break

                row_dict = dict(zip(columns, row))
                rows_data.append(row_dict)
                total_size += len(str(row_dict))
                row_count += 1

        # Convert to JSON format
        content = json.dumps(rows_data, indent=2, default=str, ensure_ascii=False)

        # Create metadata
        metadata = self._create_source_metadata(
            source_id=f"query:{query_name}",
            source_type=SourceType.DATABASE,
            content_type=ContentType.JSON,
            custom_metadata={
                "query_name": query_name,
                "row_count": len(rows_data),
                "columns": list(columns),
                "query": query_sql,
                "parameters": parameters,
                "database_type": self.database_type,
            },
        )

        return IngestionResult(
            success=True,
            source_id=f"query:{query_name}",
            content=content,
            metadata=metadata,
            raw_size_bytes=total_size,
            processed_size_bytes=len(content.encode("utf-8")),
        )

    def _find_query_config(self, query_name: str) -> Optional[Dict[str, Any]]:
        """Find query configuration by name"""
        for query_config in self.queries:
            if query_config.get("name") == query_name:
                return query_config
        return None

    async def ingest_batch(
        self, source_ids: List[str], **kwargs
    ) -> List[IngestionResult]:
        """Ingest multiple database sources"""
        results = []

        for source_id in source_ids:
            result = await self.ingest(source_id, **kwargs)
            results.append(result)

        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Database batch ingestion completed: "
            f"{successful}/{len(source_ids)} successful"
        )

        return results

    async def ingest_incremental(self, source_id: str, **kwargs) -> IngestionResult:
        """Ingest data incrementally based on timestamp or ID"""
        last_value = kwargs.get("last_value")
        timestamp_column = kwargs.get("timestamp_column", "updated_at")
        id_column = kwargs.get("id_column", "id")
        mode = kwargs.get("mode", "timestamp")  # 'timestamp' or 'id'

        if mode == "timestamp" and last_value:
            kwargs["where"] = f"{timestamp_column} > '{last_value}'"
            kwargs["order_by"] = timestamp_column
        elif mode == "id" and last_value:
            kwargs["where"] = f"{id_column} > {last_value}"
            kwargs["order_by"] = id_column

        return await self.ingest(source_id, **kwargs)

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        if not self.engine:
            await self.connect()

        if not self.engine:
            raise IngestionError("Failed to establish database connection")

        try:
            async with self.engine.begin() as conn:
                if self.database_type == "postgresql":
                    query = text(
                        """
                        SELECT
                            column_name,
                            data_type,
                            is_nullable,
                            column_default,
                            character_maximum_length
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table_name
                        ORDER BY ordinal_position
                    """
                    )
                    result = await conn.execute(
                        query, {"schema": self.schema, "table_name": table_name}
                    )
                else:
                    # Simplified schema for other databases
                    query = text(f"SELECT * FROM {table_name} LIMIT 0")
                    result = await conn.execute(query)
                    columns = result.keys()
                    return {
                        "table_name": table_name,
                        "columns": [
                            {"column_name": col, "data_type": "unknown"}
                            for col in columns
                        ],
                    }

                columns = []
                for row in result.fetchall():
                    columns.append(
                        {
                            "column_name": row[0],
                            "data_type": row[1],
                            "is_nullable": row[2],
                            "column_default": row[3],
                            "character_maximum_length": row[4],
                        }
                    )

                return {
                    "table_name": table_name,
                    "schema": self.schema,
                    "columns": columns,
                    "column_count": len(columns),
                }
        except Exception as e:
            return {"error": str(e)}

    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Get detailed information about database source"""
        try:
            source_type, source_name = source_id.split(":", 1)

            base_info = {
                "source_id": source_id,
                "connector": self.name,
                "database_type": self.database_type,
                "host": self.host,
                "database": self.database,
                "schema": self.schema,
                "last_checked": datetime.now(timezone.utc).isoformat(),
            }

            if source_type == "table":
                schema_info = await self.get_table_schema(source_name)
                base_info.update(schema_info)
            elif source_type == "query":
                query_config = self._find_query_config(source_name)
                base_info.update(
                    {"query_name": source_name, "query_config": query_config}
                )

            return base_info

        except Exception as e:
            return {"error": str(e)}


# Register the connector
ConnectorFactory.register("database", DatabaseConnector)
