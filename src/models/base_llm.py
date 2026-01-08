#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LLM Abstract Interface
Supports multiple model backends, can easily switch between different models via config file
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import yaml
import os
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """LLM Base Abstract Class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("name", "unknown")
        self.model_path = config.get("model_path", "")
        self.tokenizer_path = config.get("tokenizer_path", self.model_path)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.chat_template = config.get("chat_template", "default")
        
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def _load_model(self):
        """Load model"""
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                repetition_penalty: Optional[float] = None,
                **kwargs) -> str:
        """Generate text"""
        pass
    
    @abstractmethod
    def batch_generate(self, 
                      prompts: List[str],
                      **kwargs) -> List[str]:
        """Batch generate text"""
        pass
    
    def format_chat(self, 
                   messages: List[Dict[str, str]], 
                   chat_templates: Dict[str, Dict[str, str]]) -> str:
        """Format chat messages"""
        if self.chat_template not in chat_templates:
            # If no template specified, use simple format
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        template = chat_templates[self.chat_template]
        formatted_text = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_text += template.get("system_prefix", "") + content + template.get("system_suffix", "")
            elif role == "user":
                formatted_text += template.get("user_prefix", "") + content + template.get("user_suffix", "")
            elif role == "assistant":
                formatted_text += template.get("assistant_prefix", "") + content + template.get("assistant_suffix", "")
        
        # Add assistant start marker (for generation)
        formatted_text += template.get("assistant_prefix", "")
        return formatted_text
    
    def chat(self, 
            messages: List[Dict[str, str]], 
            chat_templates: Dict[str, Dict[str, str]],
            **kwargs) -> str:
        """Chat generation"""
        prompt = self.format_chat(messages, chat_templates)
        response = self.generate(prompt, **kwargs)
        
        # Remove prompt part, only return newly generated content
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Remove end marker
        template = chat_templates.get(self.chat_template, {})
        assistant_suffix = template.get("assistant_suffix", "")
        if assistant_suffix and response.endswith(assistant_suffix):
            response = response[:-len(assistant_suffix)].strip()
            
        return response
