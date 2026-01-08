#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node Service
Dedicated to handling node query and management
"""

from typing import List, Optional
import logging

from ...core.interfaces import INodeRetriever, NodeInfo
from ...core.exceptions import KnowledgeGraphError
from ...utils.neo4j_utils import Neo4jManager


class NodeService(INodeRetriever):
    """Node Query Service"""
    
    def __init__(self, neo4j_manager: Neo4jManager):
        """
        Initialize node service
        
        Args:
            neo4j_manager: Neo4j manager
        """
        self.neo4j_manager = neo4j_manager
        self.logger = logging.getLogger(__name__)
    
    def get_all_nodes(self, limit: int = None) -> List[NodeInfo]:
        """
        Get all nodes
        
        Args:
            limit: Limit count
            
        Returns:
            List of node information
        """
        try:
            query = "MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as properties"
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.neo4j_manager.client.execute_query(query)
            
            nodes = []
            for record in result:
                node_info = self._create_node_info(record)
                if node_info:
                    nodes.append(node_info)
            
            return nodes
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to get all nodes: {str(e)}")
    
    def get_nodes_by_label(self, label: str, limit: int = None) -> List[NodeInfo]:
        """
        Get nodes by label
        
        Args:
            label: Node label
            limit: Limit count
            
        Returns:
            List of node information
        """
        try:
            query = f"""
            MATCH (n:{label}) 
            RETURN id(n) as id, labels(n) as labels, properties(n) as properties
            """
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.neo4j_manager.client.execute_query(query)
            
            nodes = []
            for record in result:
                node_info = self._create_node_info(record)
                if node_info:
                    nodes.append(node_info)
            
            return nodes
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to get nodes by label: {str(e)}")
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """
        Get node by ID
        
        Args:
            node_id: Node ID
            
        Returns:
            Node information
        """
        try:
            # Try to query as internal ID
            try:
                query = """
                MATCH (n) WHERE id(n) = $node_id
                RETURN id(n) as id, labels(n) as labels, properties(n) as properties
                """
                result = self.neo4j_manager.client.execute_query(query, {"node_id": int(node_id)})
            except ValueError:
                # Query as property
                query = """
                MATCH (n) WHERE n.wikidata_id = $node_id OR n.id = $node_id
                RETURN id(n) as id, labels(n) as labels, properties(n) as properties
                """
                result = self.neo4j_manager.client.execute_query(query, {"node_id": node_id})
            
            for record in result:
                return self._create_node_info(record)
            
            return None
            
        except Exception as e:
            raise KnowledgeGraphError(f"Failed to get node by ID: {str(e)}")
    
    def _create_node_info(self, record) -> Optional[NodeInfo]:
        """
        Create node information from query record
        
        Args:
            record: Query record
            
        Returns:
            Node information object
        """
        try:
            properties = dict(record["properties"])
            labels = list(record["labels"])
            
            # Get node name
            name = (properties.get('name') or 
                   properties.get('wikidata_id') or 
                   f"node_{record['id']}")
            
            return NodeInfo(
                id=str(record["id"]),
                name=name,
                properties=properties,
                labels=labels
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create node info: {e}")
            return None
