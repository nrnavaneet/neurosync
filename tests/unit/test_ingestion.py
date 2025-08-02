"""
Essential tests for Phase 2 ingestion system - File, API, and Database connectors only
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from neurosync.core.config.validation import ConfigGenerator, ConfigValidator
from neurosync.core.exceptions.custom_exceptions import IngestionError
from neurosync.ingestion.base.connector import (
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceMetadata,
    SourceType,
)
from neurosync.ingestion.connectors import (
    api_connector,
    database_connector,
    file_connector,
)
from neurosync.ingestion.manager import IngestionManager


class TestConnectorFactory:
    """Test connector factory - the core of Phase 2"""

    def test_list_all_connectors(self):
        """Test all 3 Phase 2 connectors are registered (plus dummy for testing)"""
        connectors = ConnectorFactory.list_connectors()
        # We expect 3 main connectors plus the dummy one for testing
        assert len(connectors) >= 3
        expected = ["file", "api", "database"]
        for connector in expected:
            assert connector in connectors
        for connector_type in expected:
            assert connector_type in connectors

    def test_create_each_connector_type(self):
        """Test creating each connector type"""
        configs = {
            "file": {"base_path": "/tmp", "file_patterns": ["*.txt"]},
            "api": {"base_url": "https://api.example.com", "endpoints": ["/test"]},
            "database": {
                "database_type": "sqlite",
                "connection_string": "sqlite:///test.db",
            },
        }

        for connector_type, config in configs.items():
            connector = ConnectorFactory.create(connector_type, config)
            assert connector is not None


class TestFileConnector:
    """Test file connector - most commonly used"""

    def test_file_connector_initialization(self, tmp_path):
        """Test file connector setup"""
        config = {
            "base_path": str(tmp_path),
            "file_patterns": ["*.txt", "*.md"],
            "batch_size": 5,
        }
        connector = ConnectorFactory.create("file", config)

        assert str(connector.base_path) == str(tmp_path)
        assert connector.file_patterns == ["*.txt", "*.md"]
        assert connector.batch_size == 5

    @pytest.mark.asyncio
    async def test_file_connector_list_sources(self, tmp_path):
        """Test file connector can list sources"""
        # Create test files
        (tmp_path / "test1.txt").write_text("content 1")
        (tmp_path / "test2.md").write_text("# content 2")
        (tmp_path / "ignore.pdf").write_text("pdf content")

        config = {"base_path": str(tmp_path), "file_patterns": ["*.txt", "*.md"]}
        connector = ConnectorFactory.create("file", config)

        async with connector:
            sources = await connector.list_sources()
            # Should find .txt and .md files, not .pdf
            txt_files = [s for s in sources if s.endswith(".txt")]
            md_files = [s for s in sources if s.endswith(".md")]
            assert len(txt_files) >= 1
            assert len(md_files) >= 1


class TestIngestionManager:
    """Test ingestion manager - orchestrates all connectors"""

    def test_manager_initialization(self, tmp_path):
        """Test manager can initialize multiple connectors"""
        config = {
            "sources": [
                {
                    "name": "test_files",
                    "type": "file",
                    "enabled": True,
                    "config": {"base_path": str(tmp_path), "file_patterns": ["*.txt"]},
                },
                {
                    "name": "test_api",
                    "type": "api",
                    "enabled": True,
                    "config": {
                        "base_url": "https://httpbin.org",
                        "endpoints": ["/json"],
                    },
                },
            ]
        }

        manager = IngestionManager(config)
        assert len(manager.connectors) == 2
        assert "test_files" in manager.connectors
        assert "test_api" in manager.connectors

    @pytest.mark.asyncio
    async def test_manager_connection_testing(self, tmp_path):
        """Test manager can test all connections"""
        config = {
            "sources": [
                {
                    "name": "test_files",
                    "type": "file",
                    "enabled": True,
                    "config": {"base_path": str(tmp_path), "file_patterns": ["*.txt"]},
                }
            ]
        }

        manager = IngestionManager(config)

        # Mock the test_connection method
        for connector in manager.connectors.values():
            connector.test_connection = AsyncMock(return_value=True)

        results = await manager.test_connections()
        assert "test_files" in results
        assert results["test_files"] is True


class TestConfigurationSystem:
    """Test configuration validation - critical for usability"""

    def test_basic_config_generation(self):
        """Test generating basic configuration"""
        config = ConfigGenerator.generate_basic_config()
        assert "sources" in config
        assert len(config["sources"]) == 1
        assert config["sources"][0]["type"] == "file"

    def test_config_validation_success(self):
        """Test validating correct configuration"""
        valid_config = {
            "sources": [
                {
                    "name": "test_source",
                    "type": "file",
                    "enabled": True,
                    "config": {"base_path": "/tmp"},
                }
            ]
        }

        validated = ConfigValidator.validate_config(valid_config)
        assert validated is not None
        assert len(validated.sources) == 1

    def test_config_validation_errors(self):
        """Test configuration validation catches errors"""
        # Empty sources should fail
        with pytest.raises(Exception):
            ConfigValidator.validate_config({"sources": []})

        # Invalid connector type should fail
        with pytest.raises(Exception):
            ConfigValidator.validate_config(
                {"sources": [{"name": "test", "type": "invalid", "config": {}}]}
            )

    def test_config_file_roundtrip(self, tmp_path):
        """Test saving and loading configuration files"""
        config = ConfigGenerator.generate_basic_config()
        config_file = tmp_path / "test_config.json"

        # Save config
        with open(config_file, "w") as f:
            json.dump(config, f)

        # Load and validate
        loaded = ConfigValidator.load_config(str(config_file))
        validated = ConfigValidator.validate_config(loaded)

        assert validated is not None
        assert len(validated.sources) == len(config["sources"])


class TestMetadataSystem:
    """Test metadata and result objects"""

    def test_source_metadata_creation(self):
        """Test creating source metadata"""
        metadata = SourceMetadata(
            source_id="test_doc",
            source_type=SourceType.FILE,
            content_type=ContentType.TEXT,
            file_path="/tmp/test.txt",
        )

        assert metadata.source_id == "test_doc"
        assert metadata.source_type == SourceType.FILE
        assert metadata.content_type == ContentType.TEXT

    def test_ingestion_result_serialization(self):
        """Test ingestion result can be serialized"""
        result = IngestionResult(
            success=True,
            source_id="test_doc",
            content="Sample content",
            processing_time_seconds=1.5,
        )

        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["source_id"] == "test_doc"
        assert result_dict["content_length"] == len("Sample content")


class TestIntegrationWorkflow:
    """Test complete workflows end-to-end"""

    @pytest.mark.asyncio
    async def test_complete_file_ingestion_workflow(self, tmp_path):
        """Test complete file ingestion from config to results"""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document 1 content")
        (tmp_path / "doc2.txt").write_text("Document 2 content")

        # Create configuration
        config = {
            "sources": [
                {
                    "name": "test_docs",
                    "type": "file",
                    "enabled": True,
                    "config": {
                        "base_path": str(tmp_path),
                        "file_patterns": ["*.txt"],
                        "batch_size": 10,
                    },
                }
            ]
        }

        # Initialize and test manager
        manager = IngestionManager(config)
        assert len(manager.connectors) == 1

        # Test file discovery
        file_connector = manager.connectors["test_docs"]
        async with file_connector:
            sources = await file_connector.list_sources()
            assert len(sources) == 2  # Should find both .txt files


class TestFileConnectorDetailedUnittest(unittest.IsolatedAsyncioTestCase):
    """Detailed file connector tests using unittest framework"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "base_path": self.temp_dir,
            "file_patterns": ["*.txt", "*.json"],
            "batch_size": 10,
        }
        self.connector = file_connector.FileConnector(self.config)

        # Create test files
        with open(f"{self.temp_dir}/test.txt", "w") as f:
            f.write("Hello, World!")

        with open(f"{self.temp_dir}/data.json", "w") as f:
            json.dump({"message": "test data"}, f)

    async def test_connection(self):
        """Test file connector connection"""
        await self.connector.connect()
        connected = await self.connector.test_connection()
        self.assertTrue(connected)
        await self.connector.disconnect()

    async def test_list_sources(self):
        """Test listing file sources"""
        await self.connector.connect()
        sources = await self.connector.list_sources()

        # Should find our test files
        source_names = [str(Path(s).name) for s in sources]
        self.assertIn("test.txt", source_names)
        self.assertIn("data.json", source_names)

        await self.connector.disconnect()

    async def test_ingest_text_file(self):
        """Test ingesting text file"""
        await self.connector.connect()

        result = await self.connector.ingest("test.txt")

        self.assertTrue(result.success)
        self.assertEqual(result.content, "Hello, World!")
        self.assertEqual(result.metadata.content_type, ContentType.TEXT)
        self.assertGreater(result.processing_time_seconds, 0)

        await self.connector.disconnect()

    async def test_ingest_json_file(self):
        """Test ingesting JSON file"""
        await self.connector.connect()

        result = await self.connector.ingest("data.json")

        self.assertTrue(result.success)
        self.assertEqual(result.content, '{"message": "test data"}')
        self.assertEqual(result.metadata.content_type, ContentType.JSON)

        await self.connector.disconnect()

    async def test_batch_ingestion(self):
        """Test batch file ingestion"""
        await self.connector.connect()

        sources = ["test.txt", "data.json"]
        results = await self.connector.ingest_batch(sources)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.success)

        await self.connector.disconnect()


class TestAPIConnector(unittest.IsolatedAsyncioTestCase):
    """Test API connector functionality"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.config = {
            "base_url": "https://jsonplaceholder.typicode.com",
            "auth_type": "none",
            "timeout": 10,
            "endpoints": [{"path": "/posts/1", "method": "GET"}],
        }
        self.connector = api_connector.APIConnector(self.config)

    async def test_configuration_validation(self):
        """Test API connector configuration validation"""
        # Valid config should not raise
        try:
            await self.connector.connect()
            await self.connector.disconnect()
        except Exception as e:
            self.fail(f"Valid configuration raised exception: {e}")

    async def test_list_sources(self):
        """Test listing API endpoints"""
        sources = await self.connector.list_sources()
        self.assertIn("/posts/1", sources)

    @patch("neurosync.ingestion.connectors.api_connector.AsyncClient")
    async def test_mock_api_request(self, mock_client_class):
        """Test API request with mock"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"id": 1, "title": "Test Post"}
        mock_response.text = '{"id": 1, "title": "Test Post"}'
        mock_response.content = b'{"id": 1, "title": "Test Post"}'
        mock_response.url = "https://jsonplaceholder.typicode.com/posts/1"
        mock_response.headers = {"content-type": "application/json"}

        # Mock client instance
        mock_client_instance = AsyncMock()
        mock_client_instance.request.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_class.return_value = mock_client_instance

        # Test ingestion
        await self.connector.connect()
        result = await self.connector.ingest("/posts/1")

        self.assertTrue(result.success)
        self.assertIn("Test Post", result.content)

        await self.connector.disconnect()


class TestDatabaseConnector(unittest.IsolatedAsyncioTestCase):
    """Test database connector functionality with data warehouse support"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.sqlite_config = {
            "database_type": "sqlite",
            "database": ":memory:",
            "tables": ["test_table"],
        }

        self.postgres_config = {
            "database_type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "test_user",
            "password": "test_pass",
            "schema": "public",
        }

        self.snowflake_config = {
            "database_type": "snowflake",
            "account": "test-account",
            "username": "test_user",
            "password": "test_pass",
            "database": "TEST_DB",
            "warehouse": "TEST_WH",
            "schema": "PUBLIC",
            "role": "TEST_ROLE",
        }

        self.bigquery_config = {
            "database_type": "bigquery",
            "project_id": "test-project-123",
            "credentials_path": "/path/to/credentials.json",
        }

        self.redshift_config = {
            "database_type": "redshift",
            "host": "test-cluster.abc123.us-east-1.redshift.amazonaws.com",
            "port": 5439,
            "database": "test_db",
            "username": "test_user",
            "password": "test_pass",
        }

    def test_sqlite_configuration_validation(self):
        """Test SQLite connector configuration validation"""
        connector = database_connector.DatabaseConnector(self.sqlite_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.database_type, "sqlite")

    def test_postgresql_configuration_validation(self):
        """Test PostgreSQL connector configuration validation"""
        connector = database_connector.DatabaseConnector(self.postgres_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.database_type, "postgresql")

    def test_snowflake_configuration_validation(self):
        """Test Snowflake connector configuration validation"""
        connector = database_connector.DatabaseConnector(self.snowflake_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.database_type, "snowflake")
        self.assertEqual(connector.account, "test-account")
        self.assertEqual(connector.warehouse, "TEST_WH")

    def test_bigquery_configuration_validation(self):
        """Test BigQuery connector configuration validation"""
        connector = database_connector.DatabaseConnector(self.bigquery_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.database_type, "bigquery")
        self.assertEqual(connector.project_id, "test-project-123")

    def test_redshift_configuration_validation(self):
        """Test Redshift connector configuration validation"""
        connector = database_connector.DatabaseConnector(self.redshift_config)
        self.assertIsNotNone(connector)
        self.assertEqual(connector.database_type, "redshift")
        self.assertEqual(connector.port, 5439)

    def test_connection_string_building(self):
        """Test connection string building for various databases"""
        # SQLite
        sqlite_connector = database_connector.DatabaseConnector(self.sqlite_config)
        sqlite_conn_str = sqlite_connector._build_connection_string()
        self.assertIn("sqlite+aiosqlite", sqlite_conn_str)

        # PostgreSQL
        postgres_connector = database_connector.DatabaseConnector(self.postgres_config)
        postgres_conn_str = postgres_connector._build_connection_string()
        self.assertIn("postgresql+psycopg", postgres_conn_str)
        self.assertIn("localhost:5432", postgres_conn_str)

        # Snowflake - only test if available
        snowflake_connector = database_connector.DatabaseConnector(
            self.snowflake_config
        )
        try:
            snowflake_conn_str = snowflake_connector._build_connection_string()
            self.assertIn("snowflake://", snowflake_conn_str)
            self.assertIn("test-account", snowflake_conn_str)
            self.assertIn("warehouse=TEST_WH", snowflake_conn_str)
        except IngestionError as e:
            if "Snowflake connector not available" in str(e):
                self.skipTest("Snowflake connector not installed")
            else:
                raise

    def test_unsupported_database_type(self):
        """Test unsupported database type raises error"""
        invalid_config = {
            "database_type": "unsupported_db",
            "host": "localhost",
            "database": "test",
            "username": "test",
            "password": "test",
        }

        with self.assertRaises(IngestionError):
            database_connector.DatabaseConnector(invalid_config)

    def test_missing_required_fields(self):
        """Test missing required fields raise errors"""
        # Missing warehouse for Snowflake
        incomplete_snowflake = self.snowflake_config.copy()
        del incomplete_snowflake["warehouse"]

        with self.assertRaises(IngestionError):
            database_connector.DatabaseConnector(incomplete_snowflake)

        # Missing project_id for BigQuery
        incomplete_bigquery = self.bigquery_config.copy()
        del incomplete_bigquery["project_id"]

        with self.assertRaises(IngestionError):
            database_connector.DatabaseConnector(incomplete_bigquery)

    def test_supported_database_types(self):
        """Test all supported database types are recognized"""
        expected_types = {
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

        self.assertEqual(
            database_connector.DatabaseConnector.SUPPORTED_DATABASES, expected_types
        )

    async def test_test_connection_method(self):
        """Test connection testing method"""
        connector = database_connector.DatabaseConnector(self.sqlite_config)

        # Mock the engine and connection
        with patch.object(connector, "engine", create=True) as mock_engine:
            mock_conn = AsyncMock()
            mock_engine.begin.return_value.__aenter__.return_value = mock_conn

            result = await connector.test_connection()
            self.assertTrue(result)

    def test_get_test_query(self):
        """Test appropriate test queries for different databases"""
        # Standard databases
        sqlite_connector = database_connector.DatabaseConnector(self.sqlite_config)
        self.assertEqual(sqlite_connector._get_test_query(), "SELECT 1")

        # Snowflake
        snowflake_connector = database_connector.DatabaseConnector(
            self.snowflake_config
        )
        self.assertEqual(snowflake_connector._get_test_query(), "SELECT 1 as test")

        # BigQuery
        bigquery_connector = database_connector.DatabaseConnector(self.bigquery_config)
        self.assertEqual(bigquery_connector._get_test_query(), "SELECT 1 as test")


class TestSourceMetadata(unittest.TestCase):
    """Test source metadata functionality"""

    def test_metadata_creation(self):
        """Test creating source metadata"""
        metadata = SourceMetadata(
            source_id="test-source",
            source_type=SourceType.FILE,
            content_type=ContentType.TEXT,
            size_bytes=1024,
        )

        self.assertEqual(metadata.source_id, "test-source")
        self.assertEqual(metadata.source_type, SourceType.FILE)
        self.assertEqual(metadata.content_type, ContentType.TEXT)
        self.assertEqual(metadata.size_bytes, 1024)

    def test_metadata_serialization(self):
        """Test metadata to dict conversion"""
        metadata = SourceMetadata(
            source_id="test-source",
            source_type=SourceType.API,
            content_type=ContentType.JSON,
        )

        metadata_dict = metadata.to_dict()
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict["source_id"], "test-source")
        self.assertEqual(metadata_dict["source_type"], "api")
        self.assertEqual(metadata_dict["content_type"], "json")


class TestIngestionResult(unittest.TestCase):
    """Test ingestion result functionality"""

    def test_successful_result(self):
        """Test successful ingestion result"""
        result = IngestionResult(
            success=True,
            source_id="test-source",
            content="test content",
            processing_time_seconds=1.5,
            raw_size_bytes=1024,
            processed_size_bytes=512,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.source_id, "test-source")
        self.assertEqual(result.content, "test content")
        self.assertEqual(result.processing_time_seconds, 1.5)
        self.assertIsNone(result.error)

    def test_failed_result(self):
        """Test failed ingestion result"""
        result = IngestionResult(
            success=False, source_id="test-source", error="Connection failed"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Connection failed")
        self.assertIsNone(result.content)

    def test_result_serialization(self):
        """Test result to dict conversion"""
        result = IngestionResult(
            success=True, source_id="test-source", content="test content"
        )

        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["source_id"], "test-source")


class TestContentTypeDetection(unittest.TestCase):
    """Test content type detection"""

    def test_file_extension_detection(self):
        """Test detecting content type from file extensions"""
        test_cases = [
            ("document.txt", ContentType.TEXT),
            ("data.json", ContentType.JSON),
            ("report.pdf", ContentType.PDF),
            ("spreadsheet.csv", ContentType.CSV),
            ("page.html", ContentType.HTML),
            ("readme.md", ContentType.MARKDOWN),
            ("data.xml", ContentType.XML),
            ("document.docx", ContentType.DOCX),
        ]

        # Create a mock connector to test the method
        config = {"base_path": "/tmp"}
        connector = file_connector.FileConnector(config)

        for filename, expected_type in test_cases:
            detected_type = connector._detect_content_type(filename)
            self.assertEqual(
                detected_type,
                expected_type,
                f"Failed for {filename}: expected {expected_type}, got {detected_type}",
            )


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Test integration scenarios across connectors"""

    async def test_multi_connector_creation(self):
        """Test creating multiple connectors"""
        configs = {
            "file": {"base_path": "/tmp"},
            "api": {"base_url": "https://api.example.com"},
            "database": {"database_type": "sqlite", "database": ":memory:"},
        }

        connectors = {}
        for conn_type, config in configs.items():
            connectors[conn_type] = ConnectorFactory.create(conn_type, config)

        self.assertEqual(len(connectors), 3)
        self.assertIn("file", connectors)
        self.assertIn("api", connectors)
        self.assertIn("database", connectors)

    async def test_connector_context_managers(self):
        """Test async context manager support"""
        config = {"base_path": "/tmp"}

        async with file_connector.FileConnector(config) as connector:
            # Should be connected
            connected = await connector.test_connection()
            self.assertTrue(connected)

        # Should be disconnected after context exit


if __name__ == "__main__":
    unittest.main()
