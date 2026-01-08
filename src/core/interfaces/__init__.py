#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Interfaces Module
Defines the main interfaces and abstract classes used in the project
"""

from .knowledge_graph_interfaces import (
    IKnowledgeGraphRetriever,
    IKnowledgeGraphExporter,
    INodeRetriever,
    NodeInfo,
    GraphStats
)

from .implicit_interfaces import (
    IImplicitGenerator,
    IImplicitEvaluator,
    ImplicitMethod,
    ImplicitRequest,
    ImplicitPrompt
)

from .asr_interfaces import (
    IASREvaluator,
    IConstraintChecker,
    ASRResult
)

from .fluency_interfaces import (
    IFluencyEvaluator,
    FluencyResult
)

__all__ = [
    'IKnowledgeGraphRetriever',
    'IKnowledgeGraphExporter',
    'INodeRetriever',
    'NodeInfo',
    'GraphStats',
    'IImplicitGenerator',
    'IImplicitEvaluator',
    'ImplicitMethod',
    'ImplicitRequest',
    'ImplicitPrompt',
    'IASREvaluator',
    'IConstraintChecker',
    'ASRResult',
    'IFluencyEvaluator',
    'FluencyResult'
]
