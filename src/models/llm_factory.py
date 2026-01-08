#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Factory Class and Manager
Provides unified interface to create and manage different types of LLM instances
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging

from .base_llm import BaseLLM
from .openai_llm import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class LLMFactory:
    """LLM Factory Class"""
    
    _instance = None
    _config = None
    _chat_templates = None
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.load_config(config_path)
            self._initialized = True
    
    def load_config(self, config_path: Optional[str] = None):
        """Load config file"""
        if config_path is None:
            # Default config file path - pointing to config inside engineered_pipeline
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs", "models.yaml"
            )
        
        if not os.path.exists(config_path):
            # If config file doesn't exist, use default config
            logger.warning(f"Config file does not exist: {config_path}, using default config")
            return {
                "llm_providers": {
                    "openai": {
                        "provider": "openai",
                        "models": []
                    }
                }
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._config = config
        self._chat_templates = config.get("chat_templates", {})
        
        # logger.info(f"Loaded config file: {config_path}")
        # logger.info(f"Available models: {list(config.get('models', {}).keys())}")
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        if self._config is None:
            raise RuntimeError("Config not loaded, please call load_config() first")
        
        return list(self._config.get("models", {}).keys())
    
    def get_default_model(self) -> str:
        """Get default model name"""
        if self._config is None:
            raise RuntimeError("Config not loaded, please call load_config() first")
        
        return self._config.get("default_model", "")
    
    def create_llm(self, model_name: Optional[str] = None) -> BaseLLM:
        """Create LLM instance"""
        if self._config is None:
            raise RuntimeError("Config not loaded, please call load_config() first")
        
        # Use specified model or default model
        if model_name is None:
            model_name = self.get_default_model()
        
        if model_name not in self._config["models"]:
            available_models = self.get_available_models()
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        model_config = self._config["models"][model_name].copy()
        model_type = model_config.get("type", "openai").lower()
        
        # Resolve relative paths
        model_config = self._resolve_paths(model_config)
        
        logger.info(f"Creating {model_type} type LLM: {model_name}")
        
        if model_type in ("openai", "vllm_server", "openai_server"):
            return OpenAICompatibleLLM(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}, only openai is supported")
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths to absolute paths"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        for key in ["model_path", "tokenizer_path"]:
            if key in config and isinstance(config[key], str):
                path = config[key]
                if path.startswith("./"):
                    config[key] = os.path.join(base_dir, path[2:])
                elif path.startswith("~/"):
                    config[key] = os.path.expanduser(path)
        
        return config
    
    def get_chat_templates(self) -> Dict[str, Dict[str, str]]:
        """Get chat templates"""
        if self._config is None:
            raise RuntimeError("Config not loaded, please call load_config() first")
        
        return self._chat_templates


class LLMManager:
    """LLM Manager, provides high-level interface"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.factory = LLMFactory(config_path)
        self.current_llm: Optional[BaseLLM] = None
        self.current_model_name: Optional[str] = None
        
        # Try to initialize default model (safe mode)
        try:
            available_models = self.list_models()
            if available_models:
                default_model = available_models[0]
                self.switch_model(default_model)
                logger.info(f"Initialized default model: {default_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize default model: {e}")
            # Ensure attributes exist
            if not hasattr(self, 'current_llm'):
                self.current_llm = None
            if not hasattr(self, 'current_model_name'):
                self.current_model_name = None
    
    def list_models(self) -> list:
        """List all available models"""
        return self.factory.get_available_models()
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to specified model"""
        if self.current_model_name == model_name:
            # logger.info(f"Already using model {model_name}")
            return True
        
        try:
            # logger.info(f"Switching model: {self.current_model_name} -> {model_name}")
            
            # Release current model resources
            if self.current_llm is not None:
                del self.current_llm
            
            # Create new model
            self.current_llm = self.factory.create_llm(model_name)
            self.current_model_name = model_name
            
            # logger.info(f"Model switch complete: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text"""
        if self.current_llm is None:
            # Use default model
            default_model = self.factory.get_default_model()
            self.switch_model(default_model)
        
        return self.current_llm.generate(prompt, **kwargs)
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Batch generate text"""
        if self.current_llm is None:
            default_model = self.factory.get_default_model()
            self.switch_model(default_model)
        
        return self.current_llm.batch_generate(prompts, **kwargs)
    
    def chat(self, messages: list, **kwargs) -> str:
        """Chat generation"""
        if self.current_llm is None:
            default_model = self.factory.get_default_model()
            self.switch_model(default_model)
        
        chat_templates = self.factory.get_chat_templates()
        return self.current_llm.chat(messages, chat_templates, **kwargs)
    
    def get_current_model(self) -> Optional[str]:
        """Get current model name"""
        return self.current_model_name
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get current model info"""
        if self.current_llm is None:
            return None
        
        return {
            "model_name": self.current_model_name,
            "model_path": self.current_llm.model_path,
            "max_tokens": self.current_llm.max_tokens,
            "temperature": self.current_llm.temperature,
            "top_p": self.current_llm.top_p,
            "repetition_penalty": self.current_llm.repetition_penalty,
            "chat_template": self.current_llm.chat_template,
        }
