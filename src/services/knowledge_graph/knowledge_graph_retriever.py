#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Retrieval Service
Dedicated to handling knowledge graph construction and retrieval logic
"""

from typing import List, Dict, Any
import logging

from ...core.interfaces import IKnowledgeGraphRetriever, GraphStats
from ...core.exceptions import KnowledgeGraphError
from ...utils.neo4j_utils import Neo4jManager


class KnowledgeGraphRetriever(IKnowledgeGraphRetriever):
    """Knowledge Graph Retrieval Service"""
    
    def __init__(self, neo4j_manager: Neo4jManager, retry_config: dict = None):
        """
        Initialize knowledge graph retriever
        
        Args:
            neo4j_manager: Neo4j manager
            retry_config: Retry configuration dictionary, including max_retries, retry_delay, etc.
        """
        self.neo4j_manager = neo4j_manager
        self.logger = logging.getLogger(__name__)
        
        # Set default retry configuration
        if retry_config is None:
            retry_config = {
                'max_retries': 5,
                'retry_delay': 2.0,
                'wikipedia_timeout': 10
            }
        self.retry_config = retry_config
        
        # Import optional dependencies
        try:
            from ...modules.wikidata_retriever import WikidataRetriever
            # Initialize WikidataRetriever with retry configuration
            self.wikidata_retriever = WikidataRetriever(
                max_retries=retry_config.get('max_retries', 5),
                retry_delay=retry_config.get('retry_delay', 2.0)
            )
            self.has_wikidata = True
            self.logger.info(f"WikidataRetriever initialized successfully, retry config: {retry_config}")
        except ImportError:
            self.wikidata_retriever = None
            self.has_wikidata = False
            self.logger.warning("WikidataRetriever unavailable, will use Mock mode")
        except Exception as e:
            self.logger.error(f"WikidataRetriever initialization failed: {e}")
            self.wikidata_retriever = None
            self.has_wikidata = False
    
    def initialize_graph(self, seed_nodes: List[str], max_depth: int, min_sitelinks: int) -> GraphStats:
        """
        Initialize knowledge graph
        
        Args:
            seed_nodes: List of seed nodes
            max_depth: Maximum retrieval depth
            min_sitelinks: Minimum number of site links
            
        Returns:
            GraphStats: Graph statistics
        """
        try:
            if self.has_wikidata:
                # Use same mixed semantic method as test script to get more nodes
                stats = self.wikidata_retriever.initialize_graph_mixed_semantic(
                    seed_nodes, max_depth, min_sitelinks
                )
                return self._convert_to_graph_stats(stats)
            else:
                # Use Mock mode
                return self._initialize_mock_graph(seed_nodes)
                
        except Exception as e:
            raise KnowledgeGraphError(f"Knowledge graph initialization failed: {str(e)}")
    
    def get_graph_stats(self) -> GraphStats:
        """Get graph statistics"""
        try:
            stats = self.neo4j_manager.get_database_stats()
            return GraphStats(
                node_count=stats.get("node_count", 0),
                relationship_count=stats.get("relationship_count", 0),
                node_labels=stats.get("node_labels", []),
                relationship_types=stats.get("relationship_types", [])
            )
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to get graph statistics: {str(e)}")
    
    def clear_graph(self) -> bool:
        """Clear graph"""
        try:
            self.neo4j_manager.clear_database()
            return True
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to clear graph: {str(e)}")
    
    def _convert_to_graph_stats(self, stats: Dict[str, Any]) -> GraphStats:
        """Convert statistics format"""
        return GraphStats(
            node_count=stats.get("final_node_count", 0),
            relationship_count=stats.get("final_relationship_count", 0),
            node_labels=stats.get("node_labels", []),
            relationship_types=stats.get("relationship_types", [])
        )
    
    def _initialize_mock_graph(self, seed_nodes: List[str]) -> GraphStats:
        """Initialize Mock graph"""
        self.logger.info("Using Mock mode to create knowledge graph")
        
        try:
            # Create some basic Mock nodes
            mock_nodes = [
                {"id": "Q1", "name": "Medicine", "type": "Domain"},
                {"id": "Q2", "name": "Disease", "type": "Category"},
                {"id": "Q3", "name": "Drug", "type": "Category"},
                {"id": "Q4", "name": "Treatment", "type": "Category"},
                {"id": "Q5", "name": "Symptom", "type": "Category"},
            ]
            
            # Create nodes
            for node_data in mock_nodes:
                create_query = """
                CREATE (n:Entity {
                    wikidata_id: $id,
                    name: $name,
                    type: $type,
                    created_at: timestamp()
                })
                """
                self.neo4j_manager.client.execute_query(create_query, node_data)
            
            # Create relationships
            relations = [
                ("Q2", "Q1", "PART_OF"),
                ("Q3", "Q1", "PART_OF"),
                ("Q4", "Q1", "PART_OF"),
                ("Q5", "Q2", "INDICATES"),
            ]
            
            for source, target, rel_type in relations:
                rel_query = f"""
                MATCH (a:Entity {{wikidata_id: $source}})
                MATCH (b:Entity {{wikidata_id: $target}})
                CREATE (a)-[r:{rel_type}]->(b)
                """
                self.neo4j_manager.client.execute_query(rel_query, {
                    "source": source, 
                    "target": target
                })
            
            # Return Mock statistics
            return GraphStats(
                node_count=len(mock_nodes),
                relationship_count=len(relations),
                node_labels=["Entity"],
                relationship_types=["PART_OF", "INDICATES"]
            )
            
        except Exception as e:
            raise KnowledgeGraphError(f"Mock graph creation failed: {str(e)}")
