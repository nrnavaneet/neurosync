"""
Base connector interface for data ingestion
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional


class BaseConnector(ABC):
    """Base abstract class for all data connectors"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize connector with configuration"""
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is working"""
        pass

    @abstractmethod
    def extract_data(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """Extract data from source as iterator of records"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Get schema information from data source"""
        pass

    def validate_config(self) -> bool:
        """Validate connector configuration"""
        required_fields = self.get_required_config_fields()
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        return True

    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """Get list of required configuration fields"""
        pass

    def __enter__(self) -> "BaseConnector":
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit"""
        self.disconnect()
