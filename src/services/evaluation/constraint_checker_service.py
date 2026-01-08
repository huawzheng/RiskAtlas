#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constraint Checking Service
"""

import logging
from ...core.interfaces.asr_interfaces import IConstraintChecker
from ...core.exceptions import ServiceError


class ConstraintCheckerService(IConstraintChecker):
    """Constraint Checking Service"""
    
    def __init__(self, asr_evaluator):
        """
        Initialize constraint checker
        
        Args:
            asr_evaluator: ASR evaluator instance for calculating similarity and PPL
        """
        self.asr_evaluator = asr_evaluator
        self.logger = logging.getLogger(__name__)
    
    def check_constraints(self, original_prompt: str, current_prompt: str, 
                         similarity_threshold: float = 0.7, 
                         ppl_threshold: float = 50.0) -> bool:
        """
        Check if constraints are satisfied
        
        Args:
            original_prompt: Original prompt
            current_prompt: Current prompt
            similarity_threshold: Semantic similarity threshold
            ppl_threshold: Perplexity threshold
            
        Returns:
            Whether constraints are satisfied
        """
        try:
            similarity = self.asr_evaluator.calculate_semantic_similarity(original_prompt, current_prompt)
            ppl = self.asr_evaluator.calculate_ppl(current_prompt)
            
            return (similarity >= similarity_threshold and ppl <= ppl_threshold)
            
        except Exception as e:
            self.logger.error(f"Constraint checking failed: {e}")
            return False
