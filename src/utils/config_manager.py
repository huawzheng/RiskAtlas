#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager
Unified management of all pipeline configuration parameters
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database Configuration"""
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

@dataclass 
class ModelConfig:
    """Model Configuration"""
    model_name: str
    server_url: str
    max_tokens: int
    temperature: float
    top_p: float

@dataclass
class DomainConfig:
    """Domain Configuration"""
    name: str
    description: str
    test_params: Dict[str, Any]  # Test parameter configuration
    wikidata_seeds: Dict[str, Any]
    retrieval_params: Dict[str, Any]
    harm_categories: Dict[str, Any]  # Changed to Any to support new structure
    filtering_thresholds: Dict[str, Any]  # Data filtering threshold configuration
    attack_config: Dict[str, Any]  # Iterative attack configuration
    output_paths: Dict[str, str]

@dataclass
class PipelineConfig:
    """Pipeline Complete Configuration"""
    global_config: Dict[str, Any]
    database: DatabaseConfig
    models: Dict[str, ModelConfig]
    stage1: Dict[str, Any]
    stage2: Dict[str, Any]
    output: Dict[str, Any]
    performance: Dict[str, Any]
    experiment_tracking: Dict[str, Any]

class ConfigManager:
    """Configuration Manager"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Configuration file directory, defaults to configs directory of current package
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.main_config_path = self.config_dir / "pipeline_config.yaml"
        self.domains_config_dir = self.config_dir / "domains"
        
        # Ensure configuration directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.domains_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load main configuration
        self._load_main_config()
        
    def _load_main_config(self) -> None:
        """Load main configuration file"""
        try:
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                self.main_config = yaml.safe_load(f)
                logger.info(f"Successfully loaded main configuration file: {self.main_config_path}")
        except FileNotFoundError:
            logger.error(f"Main configuration file does not exist: {self.main_config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse main configuration file: {e}")
            raise
            
    def load_domain_config(self, domain_name: str) -> DomainConfig:
        """
        Load domain-specific configuration
        
        Args:
            domain_name: Domain name (e.g., medicine, finance, education)
            
        Returns:
            DomainConfig object
        """
        domain_config_path = self.domains_config_dir / f"{domain_name}.yaml"
        
        try:
            with open(domain_config_path, 'r', encoding='utf-8') as f:
                domain_data = yaml.safe_load(f)
                logger.info(f"Successfully loaded domain configuration file: {domain_config_path}")
                
                return DomainConfig(
                    name=domain_data['domain']['name'],
                    description=domain_data['domain']['description'],
                    test_params=domain_data.get('test_params', {}),
                    wikidata_seeds=domain_data['wikidata_seeds'],
                    retrieval_params=domain_data['retrieval_params'],
                    harm_categories=domain_data['harm_categories'],
                    filtering_thresholds=domain_data.get('filtering_thresholds', {}),
                    attack_config=domain_data.get('attack_config', {}),
                    output_paths=domain_data['output_paths']
                )
        except FileNotFoundError:
            logger.error(f"Domain configuration file does not exist: {domain_config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse domain configuration file: {e}")
            raise
            
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self.main_config['database']['neo4j']
        return DatabaseConfig(
            neo4j_uri=db_config['uri'],
            neo4j_user=db_config['user'], 
            neo4j_password=db_config['password']
        )
        
    def get_model_config(self, model_type: str) -> ModelConfig:
        """
        Get model configuration
        
        Args:
            model_type: Model type (harmful_prompt_generator, toxicity_evaluator, implicit)
        """
        model_config = self.main_config['models'][model_type]
        return ModelConfig(
            model_name=model_config['model_name'],
            server_url=model_config['server_url'],
            max_tokens=model_config['max_tokens'],
            temperature=model_config['temperature'],
            top_p=model_config['top_p']
        )
        
    def get_stage1_config(self) -> Dict[str, Any]:
        """Get stage 1 configuration"""
        return self.main_config['stage1']
        
    def get_stage2_config(self) -> Dict[str, Any]:
        """Get stage 2 configuration"""
        return self.main_config['stage2']
        
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.main_config['output']
        
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.main_config['performance']
        
    def get_complete_config(self, domain_name: str) -> PipelineConfig:
        """
        Get complete pipeline configuration
        
        Args:
            domain_name: Domain name
            
        Returns:
            Complete PipelineConfig object
        """
        domain_config = self.load_domain_config(domain_name)
        
        # Merge main configuration and domain configuration
        merged_config = PipelineConfig(
            global_config=self.main_config['global'],
            database=self.get_database_config(),
            models={
                'harmful_prompt_generator': self.get_model_config('harmful_prompt_generator'),
                'toxicity_evaluator': self.get_model_config('toxicity_evaluator'), 
                'implicit': self.get_model_config('implicit')
            },
            stage1=self.get_stage1_config(),
            stage2=self.get_stage2_config(),
            output=self.get_output_config(),
            performance=self.get_performance_config(),
            experiment_tracking=self.main_config['experiment_tracking']
        )
        
        return merged_config, domain_config
        
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration file
        
        Args:
            config_updates: Configuration update dictionary
        """
        # Recursively update configuration
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.main_config, config_updates)
        
        # Save updated configuration
        with open(self.main_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.main_config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info("Configuration file updated")
        
    def list_available_domains(self) -> list:
        """List all available domain configurations"""
        domain_files = list(self.domains_config_dir.glob("*.yaml"))
        return [f.stem for f in domain_files]
        
    def validate_config(self, domain_name: str) -> bool:
        """
        Validate configuration file completeness
        
        Args:
            domain_name: Domain name
            
        Returns:
            Whether validation passed
        """
        try:
            # Validate main configuration
            required_main_keys = ['global', 'database', 'models', 'stage1', 'stage2', 'output']
            for key in required_main_keys:
                if key not in self.main_config:
                    logger.error(f"Main configuration missing required key: {key}")
                    return False
                    
            # Validate domain configuration
            domain_config = self.load_domain_config(domain_name)
            if not domain_config.name:
                logger.error("Domain configuration missing name field")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

def get_config_manager() -> ConfigManager:
    """Get configuration manager instance (singleton pattern)"""
    if not hasattr(get_config_manager, 'instance'):
        get_config_manager.instance = ConfigManager()
    return get_config_manager.instance

if __name__ == "__main__":
    # Test configuration manager
    config_manager = ConfigManager()
    
    # Test loading main configuration
    print("Testing main configuration loading...")
    db_config = config_manager.get_database_config()
    print(f"Database configuration: {db_config}")
    
    # Test loading domain configuration
    print("\nTesting domain configuration loading...")
    available_domains = config_manager.list_available_domains()
    print(f"Available domains: {available_domains}")
    
    if available_domains:
        domain_name = available_domains[0]
        print(f"\nLoading {domain_name} domain configuration...")
        domain_config = config_manager.load_domain_config(domain_name)
        print(f"Domain configuration: {domain_config.name} - {domain_config.description}")
        
        # Test complete configuration
        print(f"\nGetting {domain_name} complete configuration...")
        pipeline_config, domain_config = config_manager.get_complete_config(domain_name)
        print(f"Pipeline configuration retrieved successfully")
        
        # Validate configuration
        print(f"\nValidating {domain_name} configuration...")
        is_valid = config_manager.validate_config(domain_name)
        print(f"Configuration validation result: {is_valid}")
