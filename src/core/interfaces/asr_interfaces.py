#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR (Attack Success Rate) Evaluation Related Interface Definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ASRResult:
    """ASR Evaluation Result"""
    asr_successful: bool
    target_response: str
    evaluation_reasoning: str
    is_refusal: bool
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    # Additional fields: failure analysis
    refusal_type: str = None  # keyword_triggered|intent_detected|topic_blocked|format_suspicious
    trigger_words: list = None  # List of keywords that triggered refusal
    improvement_suggestion: str = None  # Improvement suggestion


class IASREvaluator(ABC):
    """ASR Evaluator Interface"""
    
    @abstractmethod
    def check_asr(self, rewritten_prompt: str, category: str, original_prompt: str = None) -> ASRResult:
        """
        Execute attack success rate check
        
        Args:
            rewritten_prompt: Rewritten prompt
            category: Attack category
            original_prompt: Original prompt
            
        Returns:
            ASR evaluation result
        """
        pass
    
    @abstractmethod
    def calculate_semantic_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate semantic similarity"""
        pass
    
    @abstractmethod
    def calculate_ppl(self, prompt: str) -> float:
        """Calculate perplexity"""
        pass


class IConstraintChecker(ABC):
    """Constraint Checker Interface"""
    
    @abstractmethod
    def meets_constraints(self, original_prompt: str, current_prompt: str, 
                         similarity_threshold: float = 0.7, 
                         ppl_threshold: float = 50.0) -> bool:
        """Check if constraints are satisfied"""
        pass
