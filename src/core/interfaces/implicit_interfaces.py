#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implicit Attack Related Interface Definitions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ImplicitMethod(Enum):
    """Implicit Attack Method Enum"""
    BASIC = "basic"
    ENHANCED = "enhanced"


@dataclass
class ImplicitRequest:
    """Implicit Attack Request"""
    original_prompt: str
    method: ImplicitMethod
    category: str = ""
    node_name: str = ""
    semantic_card: str = ""  # Semantic card required for Enhanced method
    failure_feedback: str = ""  # Feedback from last failure, used for iterative optimization
    failed_words_list: list = None  # Accumulated list of failed keywords


@dataclass
class ImplicitPrompt:
    """Implicit Attack Prompt"""
    original_prompt: str
    implicit_prompt: str
    method: ImplicitMethod
    category: str
    ppl_score: float = 0.0  # Perplexity score
    similarity_score: float = 0.0
    prompt_id: str = ""
    node_name: str = ""


class IImplicitGenerator(ABC):
    """Implicit Attack Prompt Generator Interface"""
    
    @abstractmethod
    def generate_implicit_prompt(self, request: ImplicitRequest) -> ImplicitPrompt:
        """Generate implicit attack prompt"""
        pass
    
    @abstractmethod
    def batch_generate(self, requests: List[ImplicitRequest]) -> List[ImplicitPrompt]:
        """Batch generate implicit attack prompts"""
        pass
    
    @abstractmethod
    def get_supported_methods(self) -> List[ImplicitMethod]:
        """Get supported implicit attack methods"""
        pass


class IImplicitEvaluator(ABC):
    """Implicit Attack Effectiveness Evaluator Interface"""
    
    @abstractmethod
    def evaluate_concealment(self, original: str, implicit: str) -> float:
        """Evaluate concealment"""
        pass
    
    @abstractmethod
    def evaluate_similarity(self, original: str, implicit: str) -> float:
        """Evaluate similarity"""
        pass
