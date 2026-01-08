#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluency Evaluation Service - Based on PPL Perplexity Evaluation
"""

import math
import logging
from typing import List, Dict, Any, Optional
import os

from ...core.interfaces.fluency_interfaces import IFluencyEvaluator, FluencyResult
from ...core.exceptions import ServiceError


class FluencyEvaluatorService(IFluencyEvaluator):
    """Fluency Evaluation Service (Based on PPL Perplexity)"""
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
        Initialize fluency evaluator
        
        Args:
            model_name: Model name for calculating PPL
            device: Computing device
        """
        self.model_name = model_name
        self.device = device or "cpu"
        self.logger = logging.getLogger(__name__)
        
        # Set environment variable to avoid tokenizers parallel processing warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            # Try to import dependencies
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.has_real_models = True
            self.logger.info(f"Fluency evaluator initialized: {model_name} on {self.device}")
            
        except ImportError as e:
            self.has_real_models = False
            self.model = None
            self.tokenizer = None
            self.logger.warning(f"Dependency modules not available, will use Mock mode: {e}")
    
    def evaluate_fluency(self, text: str) -> FluencyResult:
        """
        Evaluate fluency of a single text
        
        Args:
            text: Text to evaluate
            
        Returns:
            Fluency evaluation result
        """
        try:
            if self.has_real_models:
                ppl = self._calculate_perplexity_real(text)
            else:
                ppl = self._calculate_perplexity_mock(text)
            
            # Determine if fluent (PPL < 100 is fluent)
            is_fluent = not math.isinf(ppl) and ppl <= 100.0
            
            return FluencyResult(
                text=text,
                perplexity_score=ppl,
                is_fluent=is_fluent,
                fluency_score=self._ppl_to_fluency_score(ppl),
                evaluation_method="ppl_only"
            )
            
        except Exception as e:
            raise ServiceError(f"Fluency evaluation failed: {str(e)}", "Fluency")
    
    def batch_evaluate_fluency(self, texts: List[str]) -> List[FluencyResult]:
        """
        Batch evaluate text fluency
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of fluency evaluation results
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.evaluate_fluency(text)
                results.append(result)
                
                # Log progress every 50 items
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Fluency evaluation progress: {i+1}/{len(texts)}")
                    
            except Exception as e:
                self.logger.warning(f"Text evaluation failed: {e}")
                # Add failed result
                results.append(FluencyResult(
                    text=text,
                    perplexity_score=float('inf'),
                    is_fluent=False,
                    fluency_score=0.0,
                    evaluation_method="ppl_only",
                    error_message=str(e)
                ))
        
        return results
    
    def _calculate_perplexity_real(self, text: str) -> float:
        """Calculate perplexity using real model"""
        if not text or not text.strip():
            return float('inf')
        
        try:
            import torch
            
            # Encode text
            encoded = self.tokenizer(
                text.strip(),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            input_ids = encoded["input_ids"].to(self.device)
            
            # Ensure sequence length > 1
            if input_ids.size(1) <= 1:
                return float('inf')
            
            with torch.no_grad():
                # Calculate loss
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Perplexity = exp(loss)
                perplexity = torch.exp(loss).item()
                
                return perplexity
                
        except Exception as e:
            self.logger.warning(f"PPL calculation failed: {e}")
            return float('inf')
    
    def _calculate_perplexity_mock(self, text: str) -> float:
        """Perplexity calculation in Mock mode"""
        if not text or not text.strip():
            return float('inf')
        
        # Simple simulation based on text length
        text_len = len(text.strip())
        
        # Simulate PPL: longer text tends to have more stable PPL
        if text_len < 10:
            return 80.0  # Short text has higher PPL
        elif text_len < 50:
            return 45.0  # Medium text
        else:
            return 25.0  # Long text has lower PPL
    
    def _ppl_to_fluency_score(self, ppl: float) -> float:
        """Convert PPL to fluency score between 0-1"""
        if math.isinf(ppl):
            return 0.0
        
        # Lower PPL means higher fluency
        # Based on experience: PPL < 30 (excellent), 30-60 (good), 60-100 (acceptable), >100 (not fluent)
        if ppl <= 30:
            return 1.0
        elif ppl <= 60:
            return 0.8
        elif ppl <= 100:
            return 0.5
        else:
            return max(0.0, 1.0 - math.log(ppl / 100.0))
