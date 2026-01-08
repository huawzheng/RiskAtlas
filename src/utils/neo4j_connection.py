#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Connection Factory
Provides unified Neo4j connection management, conforming to engineering standards
"""

from typing import Optional, Dict, Any
from .neo4j_utils import Neo4jManager, Neo4jConnectionConfig
from .config_manager import get_config_manager
from .logger_utils import get_logger

logger = get_logger(__name__)

class Neo4jConnectionFactory:
    """Neo4j connection factory, singleton pattern"""
    
    _instance: Optional['Neo4jConnectionFactory'] = None
    _manager: Optional[Neo4jManager] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._manager = None
    
    def get_connection(self) -> Neo4jManager:
        """
        Get Neo4j connection manager
        
        Returns:
            Neo4j manager instance
        """
        if self._manager is None or not self._manager.is_connected():
            self._create_connection()
        
        return self._manager
    
    def _create_connection(self) -> None:
        """Create Neo4j connection"""
        try:
            # Get database config from config manager
            config_manager = get_config_manager()
            db_config = config_manager.get_database_config()
            
            # Create connection config
            neo4j_config = Neo4jConnectionConfig(
                uri=db_config.neo4j_uri,
                user=db_config.neo4j_user,
                password=db_config.neo4j_password
            )
            
            # Create manager and connect
            self._manager = Neo4jManager(neo4j_config)
            
            if self._manager.connect():
                logger.info("Neo4j connection factory: Connection created successfully")
            else:
                logger.error("Neo4j connection factory: Connection creation failed")
                
        except Exception as e:
            logger.error(f"Neo4j connection factory: Error creating connection - {e}")
            self._manager = None
    
    def test_connection(self) -> bool:
        """
        Test connection status
        
        Returns:
            Whether connection is normal
        """
        try:
            manager = self.get_connection()
            if manager and manager.is_connected():
                return manager.test_connection()
            return False
        except Exception as e:
            logger.error(f"Neo4j connection factory: Connection test failed - {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Database statistics dictionary
        """
        try:
            manager = self.get_connection()
            if manager and manager.is_connected():
                return manager.get_database_stats()
            return {}
        except Exception as e:
            logger.error(f"Neo4j connection factory: Failed to get database stats - {e}")
            return {}
    
    def close_connection(self) -> None:
        """Close connection"""
        if self._manager:
            self._manager.disconnect()
            self._manager = None
            logger.info("Neo4j connection factory: Connection closed")

# Global connection factory instance
_connection_factory = Neo4jConnectionFactory()

def get_neo4j_connection() -> Neo4jManager:
    """
    Unified entry point for getting Neo4j connection
    
    Returns:
        Neo4j manager instance
    """
    return _connection_factory.get_connection()

def test_neo4j_connection() -> bool:
    """
    Unified entry point for testing Neo4j connection
    
    Returns:
        Whether connection is normal
    """
    return _connection_factory.test_connection()

def get_neo4j_stats() -> Dict[str, Any]:
    """
    Unified entry point for getting Neo4j database statistics
    
    Returns:
        Database statistics dictionary
    """
    return _connection_factory.get_database_stats()

def close_neo4j_connection() -> None:
    """Unified entry point for closing Neo4j connection"""
    _connection_factory.close_connection()

if __name__ == "__main__":
    # Test connection factory
    print("Testing Neo4j connection factory...")
    
    # Test connection
    if test_neo4j_connection():
        print("✅ Neo4j connection test succeeded")
        
        # Get statistics
        stats = get_neo4j_stats()
        print(f"📊 Database statistics: {stats}")
        
        # Get connection instance
        manager = get_neo4j_connection()
        print(f"🔗 Connection manager: {manager}")
        
    else:
        print("❌ Neo4j connection test failed")
    
    # Close connection
    close_neo4j_connection()
    print("🔒 Connection closed")
