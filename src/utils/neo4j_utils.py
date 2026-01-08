#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Database Utility Module
Provides Neo4j database connection and operation functionality
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j driver not installed, using simulation mode")

# Built-in Neo4j client class (no external dependencies)
class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        if NEO4J_AVAILABLE:
            self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        else:
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None):
        """Wrapper for client execute_query"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        return []
    
    def clear_database(self):
        """Clear database"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        return True
    
    def init_n10s_config(self):
        """Initialize n10s RDF plugin configuration"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                session.run("""
                    CALL n10s.graphconfig.init({
                        handleVocabUris: 'MAP',
                        handleMultival: 'OVERWRITE',
                        keepLangTag: false,
                        keepCustomDataTypes: false,
                        applyNeo4jNaming: false
                    })
                """)
    
    def import_rdf_from_sparql(self, sparql_query: str):
        """Import RDF data from SPARQL query"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                session.run(
                    """
                    CALL n10s.rdf.import.fetch(
                        'https://query.wikidata.org/sparql?query=' + apoc.text.urlencode($sparql),
                        'Turtle',
                        { headerParams: { Accept: 'application/x-turtle' } }
                    )
                    """,
                    sparql=sparql_query
                )
    
    def rename_node_properties(self):
        """Rename node properties, including name and description"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                # Rename name property
                session.run("""
                    MATCH (n:node)
                    SET n.name = n.node
                    REMOVE n.node
                """)
                
                # Rename description property
                session.run("""
                    MATCH (n:node)
                    WHERE n.description IS NOT NULL
                    SET n.wikipedia_description = n.description
                    REMOVE n.description
                """)
    
    def remove_isolated_nodes(self):
        """Remove isolated nodes (nodes without relationships)"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n:node)
                    WHERE NOT (n)--()
                    DELETE n
                    RETURN count(n) as deleted
                """)
                return result.single()["deleted"]
        return 0
    
    def get_node_count(self):
        """Get node count"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                return result.single()["count"]
        return 0
    
    def get_relationship_count(self):
        """Get relationship count"""
        if NEO4J_AVAILABLE and self.driver:
            with self.driver.session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                return result.single()["count"]
        return 0

from .logger_utils import get_logger

logger = get_logger(__name__)

@dataclass
class Neo4jConnectionConfig:
    """Neo4j connection configuration"""
    uri: str
    user: str
    password: str
    database: Optional[str] = None

class Neo4jManager:
    """Neo4j database manager"""
    
    def __init__(self, config: Neo4jConnectionConfig):
        """
        Initialize Neo4j manager
        
        Args:
            config: Neo4j connection configuration
        """
        self.config = config
        self.client = None
        self._connected = False
        
    def connect(self) -> bool:
        """
        Connect to Neo4j database
        
        Returns:
            Whether connection succeeded
        """
        try:
            self.client = Neo4jClient(
                uri=self.config.uri,
                user=self.config.user,
                password=self.config.password
            )
            
            # Test connection
            self.test_connection()
            self._connected = True
            logger.info(f"Successfully connected to Neo4j database: {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from database"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from Neo4j database")
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            Whether connection is normal
        """
        try:
            result = self.client.execute_query("RETURN 1 as test")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self._connected and self.client is not None
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None):
        """
        Execute Cypher query
        
        Args:
            query: Cypher query statement
            parameters: Query parameters
            
        Returns:
            Query result
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        return self.client.execute_query(query, parameters)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Database statistics dictionary
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            # Get node count
            node_count_result = self.client.execute_query("MATCH (n) RETURN count(n) as count")
            node_count = node_count_result[0]["count"] if node_count_result else 0
            
            # Get relationship count
            rel_count_result = self.client.execute_query("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_count_result[0]["count"] if rel_count_result else 0
            
            # Get node labels
            labels_result = self.client.execute_query("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            # Get relationship types
            rel_types_result = self.client.execute_query("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_types_result]
            
            stats = {
                "node_count": node_count,
                "relationship_count": rel_count,
                "node_labels": labels,
                "relationship_types": rel_types,
                "labels_count": len(labels),
                "rel_types_count": len(rel_types)
            }
            
            logger.info(f"Database statistics: {node_count} nodes, {rel_count} relationships")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear database
        
        Returns:
            Whether clearing succeeded
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            logger.warning("Clearing Neo4j database...")
            self.client.execute_query("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def get_all_nodes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all nodes
        
        Args:
            limit: Limit on number of results
            
        Returns:
            List of nodes
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            query = "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.client.execute_query(query)
            nodes = []
            
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["properties"]
                })
            
            logger.info(f"Retrieved {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []
    
    def get_nodes_by_label(self, label: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get nodes by label
        
        Args:
            label: Node label
            limit: Limit on number of results
            
        Returns:
            List of nodes
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            query = f"MATCH (n:{label}) RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.client.execute_query(query)
            nodes = []
            
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["properties"]
                })
            
            logger.info(f"Retrieved {len(nodes)} nodes with label '{label}'")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get labeled nodes: {e}")
            return []
    
    def search_nodes_by_name(self, name: str, exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Search nodes by name
        
        Args:
            name: Node name
            exact_match: Whether to use exact matching
            
        Returns:
            List of matching nodes
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            if exact_match:
                query = "MATCH (n) WHERE n.name = $name RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            else:
                query = "MATCH (n) WHERE n.name CONTAINS $name RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            
            result = self.client.execute_query(query, {"name": name})
            nodes = []
            
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["properties"]
                })
            
            logger.info(f"Search for name '{name}' found {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to search nodes: {e}")
            return []
    
    def get_node_relationships(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Get all relationships of a node
        
        Args:
            node_id: Node ID
            
        Returns:
            List of relationships
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            query = """
            MATCH (n)-[r]-(m) 
            WHERE id(n) = $node_id 
            RETURN type(r) as relationship_type, 
                   id(startNode(r)) as start_id, 
                   id(endNode(r)) as end_id,
                   properties(r) as properties,
                   properties(m) as target_properties
            """
            
            result = self.client.execute_query(query, {"node_id": node_id})
            relationships = []
            
            for record in result:
                relationships.append({
                    "type": record["relationship_type"],
                    "start_id": record["start_id"],
                    "end_id": record["end_id"],
                    "properties": record["properties"],
                    "target_properties": record["target_properties"]
                })
            
            logger.info(f"Node {node_id} has {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get node relationships: {e}")
            return []
    
    def execute_custom_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute custom query
        
        Args:
            query: Cypher query statement
            parameters: Query parameters
            
        Returns:
            Query result
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j database")
        
        try:
            result = self.client.execute_query(query, parameters or {})
            results = []
            
            for record in result:
                results.append(dict(record))
            
            logger.info(f"Custom query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

def create_neo4j_manager(config_dict: Dict[str, str]) -> Neo4jManager:
    """
    Create Neo4j manager from config dictionary
    
    Args:
        config_dict: Config dictionary containing uri, user, password
        
    Returns:
        Neo4j manager instance
    """
    config = Neo4jConnectionConfig(
        uri=config_dict['uri'],
        user=config_dict['user'],
        password=config_dict['password'],
        database=config_dict.get('database')
    )
    
    return Neo4jManager(config)

if __name__ == "__main__":
    # Test Neo4j manager
    config = Neo4jConnectionConfig(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="12345678"
    )
    
    manager = Neo4jManager(config)
    
    try:
        # Test connection
        if manager.connect():
            print("Neo4j connection test succeeded")
            
            # Get database statistics
            stats = manager.get_database_stats()
            print(f"Database statistics: {stats}")
            
            # Get some nodes
            nodes = manager.get_all_nodes(limit=5)
            print(f"Retrieved {len(nodes)} nodes")
            
        else:
            print("Neo4j connection test failed")
            
    finally:
        manager.disconnect()
