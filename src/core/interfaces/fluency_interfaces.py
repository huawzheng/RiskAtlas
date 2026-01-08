#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluency Evaluation Related Interface Definitions
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class FluencyResult:
    """Fluency Evaluation Result"""
    text: str
    perplexity_score: float
    is_fluent: bool
    fluency_score: float  # 0-1 score
    evaluation_method: str = "ppl_only"
    error_message: Optional[str] = None


class IFluencyEvaluator(ABC):
    """Fluency Evaluator Interface"""
    
    @abstractmethod
    def evaluate_fluency(self, text: str) -> FluencyResult:
        """Evaluate fluency of a single text"""
        pass
    
    @abstractmethod
    def batch_evaluate_fluency(self, texts: List[str]) -> List[FluencyResult]:
        """Batch evaluate text fluency"""
        pass
