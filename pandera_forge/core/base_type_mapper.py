"""
Abstract base class for type mapping between DataFrame types and Pandera types
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Any


class BaseTypeMapper(ABC):
    """Abstract base class for mapping DataFrame dtypes to Pandera types"""

    @abstractmethod
    def get_type_map(self) -> Dict[str, Type]:
        """
        Get the mapping dictionary from DataFrame dtype strings to Pandera types.

        Returns:
            Dictionary mapping dtype strings to Pandera type classes
        """
        pass

    @abstractmethod
    def get_pandera_type(self, dtype: Any) -> Optional[Type]:
        """
        Convert a DataFrame dtype to a Pandera type.

        Args:
            dtype: The DataFrame dtype (could be string, type object, etc.)

        Returns:
            Pandera type class or None if no mapping exists
        """
        pass

    @abstractmethod
    def normalize_dtype(self, dtype: Any) -> str:
        """
        Normalize a dtype to a standard string representation.

        Args:
            dtype: The DataFrame dtype to normalize

        Returns:
            Normalized string representation of the dtype
        """
        pass