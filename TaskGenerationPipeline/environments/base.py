"""
Base classes for environment adapters
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import StructuredTool  # ✅ FIXED


class EnvironmentAdapter(ABC):
    """
    Base class for all environment adapters
    Each environment implements this interface
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Environment-specific configuration
        """
        self.config = config or {}
        self.tools = []
        self.connection = None
        self.env_name = self.__class__.__name__
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to environment
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_tools(self) -> List[StructuredTool]:  # ✅ FIXED
        """
        Load and return tools from environment
        
        Returns:
            List of LangChain StructuredTool objects
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """
        Cleanup and close connections
        """
        pass
    
    def get_tools(self) -> List[StructuredTool]:  # ✅ FIXED
        """Get loaded tools"""
        return self.tools
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata"""
        return {
            "env_name": self.env_name,
            "num_tools": len(self.tools),
            "config": self.config
        }
