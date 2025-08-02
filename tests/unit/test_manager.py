from unittest.mock import AsyncMock, patch

import pytest

from neurosync.ingestion.manager import IngestionManager


@pytest.mark.asyncio
async def test_manager_initialization_and_ingestion():
    config = {
        "sources": [
            {"name": "dummy_source", "type": "file", "config": {"base_path": "./data"}}
        ]
    }
    with patch(
        "neurosync.ingestion.base.connector.ConnectorFactory.create"
    ) as mock_create:
        mock_connector = AsyncMock()
        mock_connector.list_sources.return_value = ["file1.txt", "file2.txt"]
        mock_connector.ingest_batch.return_value = [
            AsyncMock(success=True, source_id="file1.txt"),
            AsyncMock(success=True, source_id="file2.txt"),
        ]
        mock_connector.test_connection.return_value = True
        mock_create.return_value = mock_connector

        manager = IngestionManager(config)
        await manager.connect_all()
        conn_results = await manager.test_connections()
        assert conn_results.get("dummy_source") is True

        sources = await manager.list_all_sources()
        assert "dummy_source" in sources

        results = await manager.ingest_all_sources()
        assert "dummy_source" in results

        await manager.disconnect_all()
