#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Graph Services Module
"""

from .knowledge_graph_retriever import KnowledgeGraphRetriever
from .node_service import NodeService

# Compatibility alias
KnowledgeGraphService = KnowledgeGraphRetriever

__all__ = [
    'KnowledgeGraphRetriever',
    'NodeService',
    # Compatibility alias
    'KnowledgeGraphService'
]
