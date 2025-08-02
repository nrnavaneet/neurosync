"""
Ingestion manager for orchestrating multiple connectors
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from neurosync.core.exceptions.custom_exceptions import IngestionError
from neurosync.core.logging.logger import get_logger
from neurosync.ingestion.base.connector import (
    BaseConnector,
    ConnectorFactory,
    IngestionResult,
)


class IngestionManager:
    """Manager for orchestrating data ingestion from multiple sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.connectors: Dict[str, BaseConnector] = {}
        self.results_cache: List[IngestionResult] = []

        self._initialize_connectors()

    def _initialize_connectors(self) -> None:
        """Initialize all configured connectors"""
        sources_config = self.config.get("sources", [])

        for source_config in sources_config:
            source_name = source_config.get("name")
            connector_type = source_config.get("type")
            connector_config = source_config.get("config", {})

            if not source_name or not connector_type:
                self.logger.warning("Skipping invalid source configuration")
                continue

            try:
                connector = ConnectorFactory.create(connector_type, connector_config)
                self.connectors[source_name] = connector
                self.logger.info(
                    f"Initialized {connector_type} connector: {source_name}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize connector {source_name}: {e}")

    async def connect_all(self) -> None:
        """Connect to all configured sources"""
        tasks = []
        for name, connector in self.connectors.items():
            tasks.append(self._connect_connector(name, connector))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_connector(self, name: str, connector: BaseConnector) -> None:
        """Connect to a single connector"""
        try:
            await connector.connect()
            self.logger.info(f"Connected to {name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {name}: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect from all sources"""
        tasks = []
        for name, connector in self.connectors.items():
            tasks.append(self._disconnect_connector(name, connector))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _disconnect_connector(self, name: str, connector: BaseConnector) -> None:
        """Disconnect from a single connector"""
        try:
            await connector.disconnect()
            self.logger.info(f"Disconnected from {name}")
        except Exception as e:
            self.logger.error(f"Failed to disconnect from {name}: {e}")

    async def test_connections(self) -> Dict[str, bool]:
        """Test all connector connections"""
        results = {}
        tasks = []

        for name, connector in self.connectors.items():
            tasks.append(self._test_connector(name, connector))

        test_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(self.connectors.items()):
            result = test_results[i]
            if isinstance(result, Exception):
                results[name] = False
                self.logger.error(f"Connection test failed for {name}: {result}")
            else:
                results[name] = bool(result)

        return results

    async def _test_connector(self, name: str, connector: BaseConnector) -> bool:
        """Test connection for a single connector"""
        try:
            return await connector.test_connection()
        except Exception as e:
            self.logger.error(f"Connection test error for {name}: {e}")
            return False

    async def list_all_sources(self) -> Dict[str, List[str]]:
        """List all available sources from all connectors"""
        all_sources = {}

        for name, connector in self.connectors.items():
            try:
                sources = await connector.list_sources()
                all_sources[name] = sources
                self.logger.info(f"Found {len(sources)} sources in {name}")
            except Exception as e:
                self.logger.error(f"Failed to list sources for {name}: {e}")
                all_sources[name] = []

        return all_sources

    async def ingest_from_connector(
        self, connector_name: str, source_ids: Union[str, List[str]], **kwargs
    ) -> List[IngestionResult]:
        """Ingest data from a specific connector"""
        if connector_name not in self.connectors:
            raise IngestionError(f"Connector not found: {connector_name}")

        connector = self.connectors[connector_name]

        if isinstance(source_ids, str):
            source_ids = [source_ids]

        try:
            if len(source_ids) == 1:
                result = await connector.ingest(source_ids[0], **kwargs)
                results = [result]
            else:
                results = await connector.ingest_batch(source_ids, **kwargs)

            # Cache results
            self.results_cache.extend(results)

            # Log summary
            successful = sum(1 for r in results if r.success)
            self.logger.info(
                f"Ingestion completed for {connector_name}: "
                f"{successful}/{len(source_ids)} successful"
            )

            return results

        except Exception as e:
            self.logger.error(f"Ingestion failed for {connector_name}: {e}")
            # Return error results
            error_results = [
                IngestionResult(success=False, source_id=source_id, error=str(e))
                for source_id in source_ids
            ]
            return error_results

    async def ingest_all_sources(
        self,
        connector_filter: Optional[List[str]] = None,
        max_concurrent_connectors: int = 3,
        **kwargs,
    ) -> Dict[str, List[IngestionResult]]:
        """Ingest from all available sources"""
        # Filter connectors if specified
        connectors_to_use = {}
        if connector_filter:
            for name in connector_filter:
                if name in self.connectors:
                    connectors_to_use[name] = self.connectors[name]
        else:
            connectors_to_use = self.connectors

        if not connectors_to_use:
            self.logger.warning("No connectors to process")
            return {}

        # Get all sources first
        all_sources = await self.list_all_sources()

        # Create ingestion tasks
        semaphore = asyncio.Semaphore(max_concurrent_connectors)

        async def ingest_connector_sources(name: str) -> List[IngestionResult]:
            async with semaphore:
                sources = all_sources.get(name, [])
                if not sources:
                    return []

                return await self.ingest_from_connector(name, sources, **kwargs)

        # Run ingestion tasks
        task_names = list(connectors_to_use.keys())
        tasks = [ingest_connector_sources(name) for name in task_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results: Dict[str, List[IngestionResult]] = {}
        for i, name in enumerate(task_names):
            result = results[i]
            if isinstance(result, Exception):
                self.logger.error(f"Connector {name} failed: {result}")
                all_results[name] = []
            else:
                # Ensure we have a list of IngestionResult
                all_results[name] = result if isinstance(result, list) else []

        # Log overall summary
        total_successful = sum(
            sum(1 for r in connector_results if r.success)
            for connector_results in all_results.values()
            if isinstance(connector_results, list)
        )
        total_attempted = sum(
            len(connector_results)
            for connector_results in all_results.values()
            if isinstance(connector_results, list)
        )

        self.logger.info(
            f"Overall ingestion completed: "
            f"{total_successful}/{total_attempted} successful"
        )

        return all_results

    async def ingest_incremental(
        self, connector_name: str, last_run_file: Optional[str] = None, **kwargs
    ) -> List[IngestionResult]:
        """Perform incremental ingestion based on last run state"""
        if connector_name not in self.connectors:
            raise IngestionError(f"Connector not found: {connector_name}")

        # Load last run state for potential future use
        if last_run_file and Path(last_run_file).exists():
            try:
                with open(last_run_file, "r") as f:
                    json.load(f)  # Validate file format
                self.logger.info(f"Loaded incremental state from {last_run_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load last run state: {e}")

        # Get sources and filter for incremental processing
        connector = self.connectors[connector_name]
        all_sources = await connector.list_sources()

        # For simplicity, process all sources (in practice, you'd implement
        # timestamp-based filtering or checkpointing)
        results = await self.ingest_from_connector(
            connector_name, all_sources, **kwargs
        )

        # Save new state
        if last_run_file:
            try:
                new_state = {
                    "last_run": datetime.now(timezone.utc).isoformat(),
                    "connector": connector_name,
                    "sources_processed": len(results),
                    "successful": sum(1 for r in results if r.success),
                }

                with open(last_run_file, "w") as f:
                    json.dump(new_state, f, indent=2)

            except Exception as e:
                self.logger.error(f"Failed to save last run state: {e}")

        return results

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics from cached ingestion results"""
        if not self.results_cache:
            return {"message": "No ingestion results available"}

        successful = sum(1 for r in self.results_cache if r.success)
        failed = len(self.results_cache) - successful

        total_processing_time = sum(
            r.processing_time_seconds for r in self.results_cache
        )
        total_raw_bytes = sum(r.raw_size_bytes for r in self.results_cache)
        total_processed_bytes = sum(r.processed_size_bytes for r in self.results_cache)

        # Group by source type
        by_connector = {}
        for result in self.results_cache:
            # Extract connector type from source_id (simplified)
            connector_type = "unknown"
            if result.metadata:
                connector_type = result.metadata.source_type.value

            if connector_type not in by_connector:
                by_connector[connector_type] = {"successful": 0, "failed": 0}

            if result.success:
                by_connector[connector_type]["successful"] += 1
            else:
                by_connector[connector_type]["failed"] += 1

        return {
            "total_sources": len(self.results_cache),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / len(self.results_cache) * 100):.1f}%",
            "total_processing_time_seconds": round(total_processing_time, 2),
            "total_raw_bytes": total_raw_bytes,
            "total_processed_bytes": total_processed_bytes,
            "compression_ratio": (
                f"{(total_processed_bytes / total_raw_bytes * 100):.1f}%"
                if total_raw_bytes > 0
                else "N/A"
            ),
            "by_connector": by_connector,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear cached ingestion results"""
        self.results_cache.clear()
        self.logger.info("Cleared ingestion results cache")

    def export_results(self, file_path: str, format: str = "json") -> None:
        """Export ingestion results to file"""
        if not self.results_cache:
            raise IngestionError("No results to export")

        try:
            if format.lower() == "json":
                data = [result.to_dict() for result in self.results_cache]
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv

                with open(file_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "success",
                            "source_id",
                            "error",
                            "processing_time_seconds",
                            "raw_size_bytes",
                            "processed_size_bytes",
                        ],
                    )
                    writer.writeheader()
                    for result in self.results_cache:
                        writer.writerow(
                            {
                                "success": result.success,
                                "source_id": result.source_id,
                                "error": result.error or "",
                                "processing_time_seconds": (
                                    result.processing_time_seconds
                                ),
                                "raw_size_bytes": result.raw_size_bytes,
                                "processed_size_bytes": (result.processed_size_bytes),
                            }
                        )
            else:
                raise IngestionError(f"Unsupported export format: {format}")

            self.logger.info(
                f"Exported {len(self.results_cache)} results to {file_path}"
            )

        except Exception as e:
            raise IngestionError(f"Failed to export results: {e}")

    async def get_connector_info(self, connector_name: str) -> Dict[str, Any]:
        """Get detailed information about a connector"""
        if connector_name not in self.connectors:
            raise IngestionError(f"Connector not found: {connector_name}")

        connector = self.connectors[connector_name]

        # Get basic info
        info = {
            "name": connector_name,
            "type": connector.__class__.__name__,
            "connected": await connector.test_connection(),
        }

        # Get sources count
        try:
            sources = await connector.list_sources()
            info["sources_count"] = len(sources)
            info["sample_sources"] = sources[:5]  # First 5 sources
        except Exception as e:
            info["sources_error"] = str(e)

        return info

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect_all()
