#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Related Interface Definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class NodeInfo:
    """Node Information"""
    id: str
    name: str
    properties: Dict[str, Any]
    labels: List[str]


@dataclass
class GraphStats:
    """Graph Statistics Information"""
    node_count: int
    relationship_count: int
    node_labels: List[str]
    relationship_types: List[str]


class INodeRetriever(ABC):
    """Node Retrieval Interface"""
    
    @abstractmethod
    def get_all_nodes(self, limit: int = None) -> List[NodeInfo]:
        """Get all nodes"""
        pass
    
    @abstractmethod
    def get_nodes_by_label(self, label: str, limit: int = None) -> List[NodeInfo]:
        """Get nodes by label"""
        pass
    
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """Get node by ID"""
        pass


class IKnowledgeGraphRetriever(ABC):
    """Knowledge Graph Retrieval Interface"""
    
    @abstractmethod
    def initialize_graph(self, seed_nodes: List[str], max_depth: int, min_sitelinks: int) -> GraphStats:
        """Initialize knowledge graph"""
        pass
    
    @abstractmethod
    def get_graph_stats(self) -> GraphStats:
        """Get graph statistics"""
        pass
    
    @abstractmethod
    def clear_graph(self) -> bool:
        """Clear graph"""
        pass


class IKnowledgeGraphExporter(ABC):
    """Knowledge Graph Export Interface"""
    
    @abstractmethod
    def export_to_json(self, filename: str, domain: str) -> Dict[str, Any]:
        """Export to JSON format"""
        pass
    
    @abstractmethod
    def export_with_apoc(self, filename: str) -> Dict[str, Any]:
        """Export using APOC"""
        pass
