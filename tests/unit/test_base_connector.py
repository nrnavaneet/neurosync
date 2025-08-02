import pytest

from neurosync.core.exceptions.custom_exceptions import ConfigurationError
from neurosync.ingestion.base.connector import (
    BaseConnector,
    ConnectorFactory,
    IngestionResult,
)


class DummyConnector(BaseConnector):
    def _validate_config(self):
        pass

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def test_connection(self):
        return True

    async def list_sources(self):
        return ["src1", "src2"]

    async def ingest(self, source_id, **kwargs):
        return IngestionResult(success=True, source_id=source_id)

    async def ingest_batch(self, source_ids, **kwargs):
        return [IngestionResult(success=True, source_id=s) for s in source_ids]


def test_connector_factory_registration_and_creation():
    ConnectorFactory.register("dummy", DummyConnector)
    connector = ConnectorFactory.create("dummy", {})
    assert isinstance(connector, DummyConnector)

    with pytest.raises(ConfigurationError):
        ConnectorFactory.create("unknown", {})


@pytest.mark.asyncio
async def test_dummy_connector_methods():
    connector = DummyConnector({})
    await connector.connect()
    assert await connector.test_connection() is True
    sources = await connector.list_sources()
    assert "src1" in sources
    result = await connector.ingest("src1")
    assert result.success is True
    batch_results = await connector.ingest_batch(["src1", "src2"])
    assert len(batch_results) == 2
