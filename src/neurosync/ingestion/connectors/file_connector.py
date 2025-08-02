"""
File system connector for local files and directories
"""

import asyncio
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
import chardet

from neurosync.core.exceptions.custom_exceptions import (
    ConfigurationError,
    IngestionError,
)
from neurosync.ingestion.base.connector import (
    BaseConnector,
    ConnectorFactory,
    ContentType,
    IngestionResult,
    SourceType,
)

# import magic  # Commenting out due to system dependency


class FileConnector(BaseConnector):
    """Connector for local file system ingestion"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("base_path", "."))
        self.file_patterns = config.get("file_patterns", ["*.*"])
        self.supported_extensions = config.get(
            "supported_extensions",
            [".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".xml", ".html", ".htm"],
        )
        self.batch_size = config.get("batch_size", 100)
        self.max_file_size = (
            config.get("max_file_size_mb", 100) * 1024 * 1024
        )  # Convert to bytes
        self.recursive = config.get("recursive", True)
        self.follow_symlinks = config.get("follow_symlinks", False)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate file connector configuration"""
        if not self.base_path:
            raise ConfigurationError("base_path is required for file connector")

        # Create directory if it doesn't exist (for testing scenarios)
        base_path = Path(self.base_path)
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created base path directory: {self.base_path}")
            except Exception as e:
                self.logger.warning(f"Could not create base path {self.base_path}: {e}")

        if not self.file_patterns:
            raise ConfigurationError("At least one file_pattern is required")

        # Validate file patterns
        for pattern in self.file_patterns:
            if not isinstance(pattern, str):
                raise ConfigurationError(f"Invalid file pattern: {pattern}")

        # Validate batch size
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be greater than 0")

        # Validate max file size
        if self.max_file_size <= 0:
            raise ConfigurationError("max_file_size must be greater than 0")

    async def connect(self) -> None:
        """Establish connection (no-op for file system)"""
        self.logger.info(f"Connected to file system at: {self.base_path}")

    async def disconnect(self) -> None:
        """Close connection (no-op for file system)"""
        self.logger.info("Disconnected from file system")

    async def test_connection(self) -> bool:
        """Test if base path is accessible"""
        try:
            return self.base_path.exists() and self.base_path.is_dir()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    async def list_sources(self) -> List[str]:
        """List all supported files in the base path"""
        sources = []

        try:
            pattern = "**/*" if self.recursive else "*"
            for file_path in self.base_path.glob(pattern):
                if self._is_valid_file(file_path):
                    relative_path = file_path.relative_to(self.base_path)
                    sources.append(str(relative_path))
        except Exception as e:
            raise IngestionError(f"Failed to list sources: {e}")

        self.logger.info(f"Found {len(sources)} valid files")
        return sources

    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for ingestion"""
        if not file_path.is_file():
            return False

        # Check if it's a symlink and we're not following them
        if file_path.is_symlink() and not self.follow_symlinks:
            return False

        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False

        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                self.logger.warning(f"File too large: {file_path}")
                return False
        except OSError:
            return False

        return True

    async def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            async with aiofiles.open(file_path, "rb") as f:
                raw_data = await f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get("encoding", "utf-8")
        except Exception:
            return "utf-8"

    async def _extract_text_content(
        self, file_path: Path, content_type: ContentType
    ) -> str:
        """Extract text content based on file type"""
        try:
            if content_type == ContentType.PDF:
                return await self._extract_pdf_text(file_path)
            elif content_type == ContentType.DOCX:
                return await self._extract_docx_text(file_path)
            elif content_type in [
                ContentType.TEXT,
                ContentType.MARKDOWN,
                ContentType.JSON,
                ContentType.CSV,
                ContentType.XML,
                ContentType.HTML,
            ]:
                encoding = await self._detect_encoding(file_path)
                async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                    return await f.read()
            else:
                raise IngestionError(f"Unsupported content type: {content_type}")
        except Exception as e:
            raise IngestionError(f"Failed to extract text from {file_path}: {e}")

    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2

            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            # Run PDF extraction in thread pool to avoid blocking
            def extract_pdf():
                import io

                reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_pdf)
            return text.strip()

        except ImportError:
            raise IngestionError("PyPDF2 not installed. Run: pip install PyPDF2")
        except Exception as e:
            raise IngestionError(f"PDF extraction failed: {e}")

    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document

            def extract_docx():
                doc = Document(file_path)
                text = []
                for paragraph in doc.paragraphs:
                    text.append(paragraph.text)
                return "\n".join(text)

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, extract_docx)
            return text.strip()

        except ImportError:
            raise IngestionError(
                "python-docx not installed. Run: pip install python-docx"
            )
        except Exception as e:
            raise IngestionError(f"DOCX extraction failed: {e}")

    async def ingest(self, source_id: str, **kwargs) -> IngestionResult:
        """Ingest a single file"""
        start_time = asyncio.get_event_loop().time()
        file_path = self.base_path / source_id

        try:
            if not self._is_valid_file(file_path):
                return IngestionResult(
                    success=False,
                    source_id=source_id,
                    error="File is not valid for ingestion",
                )

            # Get file stats
            stat = file_path.stat()
            content_type = self._detect_content_type(str(file_path))

            # Extract content
            content = await self._extract_text_content(file_path, content_type)

            # Create metadata
            metadata = self._create_source_metadata(
                source_id=source_id,
                source_type=SourceType.FILE,
                content_type=content_type,
                file_path=str(file_path),
                size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                encoding=await self._detect_encoding(file_path),
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            result = IngestionResult(
                success=True,
                source_id=source_id,
                content=content,
                metadata=metadata,
                processing_time_seconds=processing_time,
                raw_size_bytes=stat.st_size,
                processed_size_bytes=len(content.encode("utf-8")),
            )

            self.logger.info(
                f"Successfully ingested file: {source_id} ({stat.st_size} bytes)"
            )
            return result

        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Failed to ingest file {source_id}: {e}")
            return IngestionResult(
                success=False,
                source_id=source_id,
                error=str(e),
                processing_time_seconds=processing_time,
            )

    async def ingest_batch(
        self, source_ids: List[str], **kwargs
    ) -> List[IngestionResult]:
        """Ingest multiple files concurrently"""
        max_concurrent = kwargs.get("max_concurrent", 5)

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)

        async def ingest_with_semaphore(source_id: str):
            async with semaphore:
                return await self.ingest(source_id, **kwargs)

        # Run ingestion tasks concurrently
        tasks = [ingest_with_semaphore(source_id) for source_id in source_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    IngestionResult(
                        success=False, source_id=source_ids[i], error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.success)
        self.logger.info(
            f"Batch ingestion completed: {successful}/{len(source_ids)} successful"
        )

        return processed_results

    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Get detailed information about a file"""
        file_path = self.base_path / source_id

        if not file_path.exists():
            return {"error": "File not found"}

        try:
            stat = file_path.stat()
            content_type = self._detect_content_type(str(file_path))

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))

            return {
                "source_id": source_id,
                "connector": self.name,
                "file_path": str(file_path),
                "size_bytes": stat.st_size,
                "size_human": self._format_bytes(stat.st_size),
                "content_type": content_type.value,
                "mime_type": mime_type,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_readable": file_path.is_file()
                and file_path.stat().st_size <= self.max_file_size,
                "last_checked": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes to human readable string"""
        bytes_float = float(bytes_value)
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_float < 1024.0:
                return f"{bytes_float:.1f} {unit}"
            bytes_float /= 1024.0
        return f"{bytes_float:.1f} TB"


# Register the connector
ConnectorFactory.register("file", FileConnector)
