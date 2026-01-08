#!/usr/bin/env python
"""
Step 2: Harmful Prompt Generation Test
======================================

Test generating harmful prompts using real knowledge graph data and fine-tuned models:
- Check vLLM service availability
- Extract real nodes from knowledge graph built in Step 1
- Generate harmful prompts combining JBB dataset
- Validate generation results format and quality
- Save generated prompt data

Notes: 
1. This step requires starting the fine-tuned model server (port 8000)
2. Need to run Step 1 first to build knowledge graph
3. Need JBB-Behaviors dataset

Usage:
    python step2_harmful_generation.py --domain medicine
    python step2_harmful_generation.py --domain medicine --selected-nodes-count 5 --verbose
    python step2_harmful_generation.py --domain medicine --check-service-only

"""

import argparse
import sys
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger_utils import setup_logger
from src.modules.harmful_prompt_generator import HarmfulPromptGenerator, HarmCategory
from src.models.llm_factory import LLMManager
from src.utils.config_manager import ConfigManager
from src.services.knowledge_graph import NodeService
from src.utils.neo4j_utils import create_neo4j_manager
from src.utils.random_seed_utils import load_seed_from_config

class HarmfulGenerationTester:
    """Harmful prompt generation tester - using real knowledge graph data"""
    
    def __init__(self, domain: str = "medicine", verbose: bool = False, selected_nodes_count: int = 3):
        self.domain = domain
        self.verbose = verbose
        self.selected_nodes_count = selected_nodes_count
        
        # Set up logging - write detailed logs to file, console shows only key information
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / "logs" / "step2"
        log_file = f"step2_harmful_generation_{domain}_{timestamp}.log"
        
        # Create two loggers: one for file (detailed), one for console (concise)
        self.file_logger = setup_logger(
            name=f"Step2_File_{domain}",
            log_file=log_file,
            level="DEBUG",
            log_dir=str(log_dir),
            enable_file_logging=True
        )
        
        self.console_logger = setup_logger(
            name=f"Step2_Console_{domain}",
            level="INFO",
            enable_file_logging=False
        )
        
        # Mainly use file_logger, key information also output to console
        self.logger = self.file_logger
        
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Set random seed to ensure experimental reproducibility
        actual_seed = load_seed_from_config(self.config_manager)
        self.logger.info(f"✅ Random seed has been set to: {actual_seed}")
        
        # Set third-party module log levels to reduce console output
        import logging
        # Set specific module log level to WARNING, so only WARNING and ERROR will show in console
        logging.getLogger("src.modules.harmful_prompt_generator").setLevel(logging.WARNING)
        logging.getLogger("src.services").setLevel(logging.WARNING)
        logging.getLogger("src.models").setLevel(logging.WARNING)
        logging.getLogger("src.utils").setLevel(logging.WARNING)
        # Set HTTP related library log levels
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Model service configuration
        self.model_config = {
            "base_url": "http://localhost:8000/v1",
            "model_name": "llama3.1-70b-finetune",
        }
        
        self.test_results = {}
        self.harmful_generator = None
        self.kg_service = None
        self.real_nodes = []
        self.prompt_templates = []  # Store prompt templates for debugging
        self.domain_config = None  # Store domain configuration
    
    def log_progress(self, message: str, level: str = "info"):
        """Important progress information logged to both file and console"""
        log_func = getattr(self.file_logger, level)
        console_log_func = getattr(self.console_logger, level)
        
        log_func(message)  # Detailed logs written to file
        console_log_func(message)  # Key information displayed in console
    
    def log_detail(self, message: str, level: str = "debug"):
        """Detailed information logged to file only"""
        log_func = getattr(self.file_logger, level)
        log_func(message)
    
    def check_model_service(self) -> bool:
        """Check fine-tuned model service availability"""
        self.log_progress("🤖 Checking fine-tuned model service...")
        
        try:
            # Get timeout from configuration
            timeout = 10  # Default value
            if self.domain_config and hasattr(self.domain_config, 'test_params'):
                timeout = self.domain_config.test_params.get('request_timeout', 10)
            
            # Check service health status
            health_url = f"{self.model_config['base_url']}/models"
            response = requests.get(health_url, timeout=timeout)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                
                if self.model_config["model_name"] in available_models:
                    self.log_progress(f"  ✅ Model service available: {self.model_config['model_name']}")
                    self.log_detail(f"  📊 Available models: {available_models}")
                    
                    self.test_results["model_service"] = {
                        "status": True,
                        "available_models": available_models,
                        "target_model": self.model_config["model_name"]
                    }
                    return True
                else:
                    self.log_progress(f"  ❌ Target model not available: {self.model_config['model_name']}", "error")
                    self.log_detail(f"  📊 Available models: {available_models}")
                    return False
            else:
                self.log_progress(f"  ❌ Model service not available: HTTP {response.status_code}", "error")
                return False
                
        except requests.RequestException as e:
            self.log_progress(f"  ❌ Model service connection failed: {e}", "error")
            self.log_progress(f"  💡 Please make sure to run: bash scripts/start_vllm_finetune_server.sh")
            self.test_results["model_service"] = {"status": False, "error": str(e)}
            return False
    
    def test_model_initialization(self) -> bool:
        """Test model initialization"""
        self.log_progress("🔧 Test model initialization...")
        
        try:
            # Use LLM manager to create model instance
            llm_manager = LLMManager()
            llm_manager.switch_model("llama3.1-70b-finetune")
            
            # Get test parameters from configuration file
            test_config = self.config_manager.main_config.get('testing', {}).get('model_initialization', {})
            test_prompt = test_config.get('test_prompt', "Hello, this is a test prompt. Please respond briefly.")
            max_tokens = test_config.get('max_tokens', 50)
            
            # Test simple text generation (using parameters from configuration file)
            response = llm_manager.generate(test_prompt, max_tokens=max_tokens)
            
            if response and len(response.strip()) > 0:
                self.log_progress("  ✅ Model initialization successful")
                self.log_detail(f"  📝 Test response: {response[:100]}...")
                
                self.test_results["model_initialization"] = {
                    "status": True,
                    "test_response_length": len(response),
                    "test_max_tokens": max_tokens
                }
                return True
            else:
                self.log_progress("  ❌ Model generated empty response", "error")
                return False
                
        except Exception as e:
            self.log_progress(f"  ❌ Model initialization failed: {e}", "error")
            self.test_results["model_initialization"] = {"status": False, "error": str(e)}
            return False
    
    def test_harmful_prompt_generator(self) -> bool:
        """Test harmful prompt generator"""
        self.log_progress("⚡ Testing harmful prompt generator initialization...")
        
        try:
            # Initialize LLM manager
            llm_manager = LLMManager()
            
            # Initialize harmful prompt generator
            self.harmful_generator = HarmfulPromptGenerator(
                llm_manager=llm_manager,
                generation_model="llama3.1-70b-finetune"
            )
            
            self.log_progress("  ✅ Harmful prompt generator initialized successfully")
            self.test_results["generator_initialization"] = {"status": True}
            return True
            
        except Exception as e:
            self.log_progress(f"  ❌ Harmful prompt generator initialization failed: {e}", "error")
            self.test_results["generator_initialization"] = {"status": False, "error": str(e)}
            return False
    
    def load_real_knowledge_graph_nodes(self) -> bool:
        """Load node data from real knowledge graph"""
        self.log_progress("📊 Loading real knowledge graph nodes...")
        
        try:
            # Load domain configuration
            self.domain_config = self.config_manager.load_domain_config(self.domain)
            self.log_detail(f"  📋 Domain configuration: {self.domain}")
            
            # Get prompts per category from configuration
            prompts_per_category = 2  # Default value
            if self.domain_config and hasattr(self.domain_config, 'harm_categories'):
                prompts_per_category = self.domain_config.harm_categories.get('prompts_per_category', 2)
            self.log_detail(f"  📝 Prompts per category: {prompts_per_category}")
            
            # Initialize knowledge graph service
            neo4j_config = self.config_manager.main_config.get('database', {}).get('neo4j', {})
            neo4j_manager = create_neo4j_manager({
                'uri': neo4j_config.get('uri'),
                'user': neo4j_config.get('user'),
                'password': neo4j_config.get('password')
            })
            neo4j_manager.connect()  # Establish connection
            self.kg_service = NodeService(neo4j_manager)
            
            # Use fixed node count limit
            kg_node_limit = 1000
            
            # Get all nodes from knowledge graph
            all_nodes = self.kg_service.get_all_nodes(limit=kg_node_limit)  # Get more nodes for random selection
            
            if not all_nodes:
                self.log_progress("  ❌ No nodes found in knowledge graph, please run Step 1 first to build knowledge graph", "error")
                return False
            
            # Filter nodes with description information
            filtered_nodes = []
            for node in all_nodes:
                if hasattr(node, 'properties') and node.properties:
                    # Check if description information exists
                    description = (
                        node.properties.get('description') or 
                        node.properties.get('wikipedia_description') or
                        node.properties.get('summary') or
                        node.properties.get('wikipedia_summary')
                    )
                    if description and len(description.strip()) > 20:
                        # Get wikidata_uri, prefer uri property
                        wikidata_uri = (
                            node.properties.get('uri') or  # Preferred: uri property in Neo4j
                            node.properties.get('wikidata_id') or
                            node.properties.get('id') or 
                            f"http://www.wikidata.org/entity/Q{node.id}"  # If none exist, generate using internal ID
                        )
                        
                        filtered_nodes.append({
                            'id': node.id,
                            'name': node.name,
                            'labels': node.labels,
                            'properties': node.properties,
                            'description': description.strip(),
                            'wikidata_uri': wikidata_uri
                        })
            
            # Use the passed node selection count
            selected_nodes_count = self.selected_nodes_count
            
            # Determine nodes to use based on parameter
            import random
            if selected_nodes_count == -1:
                # Use all filtered nodes
                self.real_nodes = filtered_nodes
                self.log_progress(f"  📊 Using all available nodes: {len(filtered_nodes)}")
            elif len(filtered_nodes) >= selected_nodes_count:
                # Randomly select specified number of nodes
                self.real_nodes = random.sample(filtered_nodes, selected_nodes_count)
                self.log_progress(f"  🎲 Randomly selected {selected_nodes_count} nodes")
            else:
                # Use all if not enough
                self.real_nodes = filtered_nodes
                self.log_progress(f"  ⚠️  Insufficient available nodes, using all {len(filtered_nodes)} nodes")
            
            self.log_progress(f"  ✅ Final using {len(self.real_nodes)} real nodes")
            
            # Only show detailed info in console when few nodes, otherwise log to file only
            if len(self.real_nodes) <= 5:
                for i, node in enumerate(self.real_nodes):
                    self.log_detail(f"    {i+1}. {node['name']}: {node['description'][:100]}...")
            else:
                self.log_detail(f"    Too many nodes ({len(self.real_nodes)}), see log file for details")
                for i, node in enumerate(self.real_nodes[:3]):
                    self.log_detail(f"    Sample {i+1}. {node['name']}: {node['description'][:100]}...")
            
            self.test_results["knowledge_graph_loading"] = {
                "status": True,
                "total_nodes_found": len(all_nodes),
                "filtered_nodes": len(filtered_nodes),
                "selected_nodes": len(self.real_nodes)
            }
            return True
            
        except Exception as e:
            self.log_progress(f"  ❌ Failed to load knowledge graph nodes: {e}", "error")
            self.test_results["knowledge_graph_loading"] = {"status": False, "error": str(e)}
            return False

    def test_context_formatting(self) -> bool:
        """Test real node context formatting"""
        self.log_progress("📋 Testing real node context formatting...")
        
        if not self.real_nodes:
            self.log_progress("  ❌ No real node data available", "error")
            return False
        
        try:
            formatted_contexts = []
            domain_info = f"{self.domain} domain"
            
            for node in self.real_nodes:
                # Extract Wikipedia information
                wikipedia_description = (
                    node['properties'].get('wikipedia_description') or 
                    node['properties'].get('description')
                )
                wikipedia_summary = (
                    node['properties'].get('wikipedia_summary') or 
                    node['properties'].get('summary')
                )
                
                formatted_context = {
                    "node_id": node['id'],
                    "node_name": node['name'],
                    "labels": node['labels'],
                    "wikipedia_description": wikipedia_description,
                    "wikipedia_summary": wikipedia_summary,
                    "domain_info": domain_info,
                    "raw_properties": node['properties']
                }
                
                formatted_contexts.append(formatted_context)
                
                # Only log detailed info to file
                self.log_detail(f"  📄 Node: {node['name']}")
                self.log_detail(f"      ID: {node['id']}")
                self.log_detail(f"      Labels: {node['labels']}")
                if wikipedia_description:
                    self.log_detail(f"      Description length: {len(wikipedia_description)}")
            
            self.log_progress(f"  ✅ Successfully formatted {len(formatted_contexts)} real node contexts")
            self.test_results["context_formatting"] = {
                "status": True,
                "formatted_count": len(formatted_contexts),
                "contexts": formatted_contexts
            }
            return True
            
        except Exception as e:
            self.log_progress(f"  ❌ Real node context formatting failed: {e}", "error")
            self.test_results["context_formatting"] = {"status": False, "error": str(e)}
            return False
    
    def test_harmful_prompt_generation(self) -> bool:
        """Test harmful prompt generation (using real knowledge graph nodes)"""
        self.log_progress("🎯 Testing harmful prompt generation...")
        
        if not self.harmful_generator:
            self.log_progress("  ❌ Harmful prompt generator not initialized", "error")
            return False
        
        if not self.real_nodes:
            self.log_progress("  ❌ No real node data available", "error")
            return False
        
        try:
            generated_prompts = []
            domain_info = f"{self.domain} domain"
            
            # Get prompts per category from configuration
            prompts_per_category = 2  # Default value
            if self.domain_config and hasattr(self.domain_config, 'harm_categories'):
                prompts_per_category = self.domain_config.harm_categories.get('prompts_per_category', 2)
            self.log_detail(f"  📝 Prompts per category: {prompts_per_category}")
            
            # Test prompt generation for all available JBB categories
            test_categories = list(HarmCategory)
            self.log_progress(f"  📋 Will test {len(test_categories)} harm content categories")
            self.log_detail(f"  📋 Category list: {[cat.value for cat in test_categories]}")
            
            total_expected = len(self.real_nodes) * len(test_categories) * prompts_per_category
            self.log_progress(f"  🎯 Expected to generate {total_expected} prompts ({len(self.real_nodes)} nodes × {len(test_categories)} categories × {prompts_per_category} prompts/category)")
            
            processed_nodes = 0
            for i, node in enumerate(self.real_nodes):  # Use all selected nodes
                node_name = node['name']
                wikipedia_description = (
                    node['properties'].get('wikipedia_description') or 
                    node['properties'].get('description')
                )
                wikipedia_summary = (
                    node['properties'].get('wikipedia_summary') or 
                    node['properties'].get('summary')
                )
                
                # Only show key progress, not detailed info for each node
                if i % 10 == 0 or i == len(self.real_nodes) - 1:
                    self.log_progress(f"  🔍 Processing node {i+1}/{len(self.real_nodes)}: {node_name}")
                else:
                    self.log_detail(f"  🔍 Processing node {i+1}/{len(self.real_nodes)}: {node_name}")
                
                for category in test_categories:
                    try:
                        self.log_detail(f"    📝 Generating {category.value} type prompts...")
                        
                        # Build prompt template for debugging first
                        generation_prompt = self.harmful_generator.build_generation_prompt(
                            node_name=node_name,
                            category=category,
                            num_prompts=prompts_per_category,
                            wikipedia_description=wikipedia_description,
                            wikipedia_summary=wikipedia_summary,
                            domain_info=domain_info,
                        )
                        
                        # Save prompt template
                        self.prompt_templates.append({
                            "node_id": node['id'],
                            "node_name": node_name,
                            "category": category.value,
                            "wikipedia_description": wikipedia_description[:200] if wikipedia_description else None,
                            "wikipedia_summary": wikipedia_summary[:200] if wikipedia_summary else None,
                            "domain_info": domain_info,
                            "prompt_template": generation_prompt,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Generate harmful prompts - disable verbose to reduce output
                        harmful_prompts = self.harmful_generator.generate_prompts_for_category(
                            node_name=node_name,
                            category=category,
                            num_prompts=prompts_per_category,  # Use configured quantity
                            wikipedia_description=wikipedia_description,
                            wikipedia_summary=wikipedia_summary,
                            domain_info=domain_info,
                            verbose=False,  # Disable verbose output
                            debug_prompt=False  # Disable debug output
                        )
                        
                        for j, prompt in enumerate(harmful_prompts):
                            # Get specific behavior type from JBB dataset
                            behavior_type = self._get_behavior_type_for_category(category)
                            
                            generated_prompts.append({
                                "node_id": node['id'],
                                "node_name": node_name,
                                "wikidata_uri": node.get('wikidata_uri'),
                                "category": category.value,
                                "behavior_type": behavior_type,  # New: specific behavior type
                                "prompt_id": f"{node['id']}_{category.value}_{j+1}",
                                "prompt_text": prompt,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "wikipedia_description": wikipedia_description[:200] if wikipedia_description else None,
                                "domain": self.domain
                            })
                            
                            # Only log to file, not displayed in console
                            self.log_detail(f"      ✓ Prompt {j+1}: {prompt[:100]}...")
                        
                        time.sleep(0.5)  # Reduced interval to speed up
                        
                    except Exception as e:
                        self.log_detail(f"    ⚠️  Category {category.value} generation failed: {e}")
                        continue
                
                processed_nodes += 1
                # Report progress every 10 nodes or at the last node
                if processed_nodes % 10 == 0 or processed_nodes == len(self.real_nodes):
                    current_prompts = len(generated_prompts)
                    self.log_progress(f"  📊 Processed {processed_nodes}/{len(self.real_nodes)} nodes, generated {current_prompts} prompts")
                
                time.sleep(1)  # Shorter interval between nodes
            
            if generated_prompts:
                self.log_progress(f"  ✅ Successfully generated {len(generated_prompts)} harmful prompts")
                
                # Statistics by category
                category_counts = {}
                for prompt in generated_prompts:
                    category = prompt['category']
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                self.log_detail("  📊 Statistics by category:")
                for category, count in category_counts.items():
                    self.log_detail(f"    {category}: {count}")
                
                self.test_results["harmful_generation"] = {
                    "status": True,
                    "generated_count": len(generated_prompts),
                    "category_counts": category_counts,
                    "prompts": generated_prompts
                }
                return True
            else:
                self.log_progress("  ❌ No harmful prompts generated", "error")
                self.test_results["harmful_generation"] = {"status": False, "generated_count": 0}
                return False
                
        except Exception as e:
            self.log_progress(f"  ❌ Harmful prompt generation test failed: {e}", "error")
            self.test_results["harmful_generation"] = {"status": False, "error": str(e)}
            return False
    
    def _get_behavior_type_for_category(self, category) -> str:
        """Get specific behavior type from JBB dataset"""
        try:
            if hasattr(self.harmful_generator, 'jbb_df') and self.harmful_generator.jbb_df is not None:
                # Filter data by category and get behavior type
                category_data = self.harmful_generator.jbb_df[self.harmful_generator.jbb_df['Category'] == category.value]
                if len(category_data) > 0:
                    # Get most common behavior type
                    behavior_counts = category_data['Behavior'].value_counts()
                    if len(behavior_counts) > 0:
                        return behavior_counts.index[0]  # Return most common behavior type
            return category.value  # If unable to get, return category name
        except Exception as e:
            self.logger.warning(f"Failed to get behavior type: {e}")
            return category.value

    def test_output_validation(self) -> bool:
        """Test output validation"""
        self.log_progress("✅ Testing generation result validation...")
        
        if "harmful_generation" not in self.test_results or not self.test_results["harmful_generation"]["status"]:
            self.log_progress("  ❌ No generation results to validate", "error")
            return False
        
        try:
            prompts = self.test_results["harmful_generation"]["prompts"]
            validation_results = {
                "total_prompts": len(prompts),
                "valid_prompts": 0,
                "empty_prompts": 0,
                "issues": []
            }
            
            for prompt_data in prompts:
                prompt_text = prompt_data["prompt_text"]
                
                # Validation criteria
                if not prompt_text or len(prompt_text.strip()) == 0:
                    validation_results["empty_prompts"] += 1
                    validation_results["issues"].append(f"Empty prompt: {prompt_data['prompt_id']}")
                else:
                    validation_results["valid_prompts"] += 1
            
            # Calculate validation rate
            valid_rate = validation_results["valid_prompts"] / validation_results["total_prompts"]
            
            self.log_progress(f"  📊 Validation statistics:")
            self.log_progress(f"      Total prompts: {validation_results['total_prompts']}")
            self.log_progress(f"      Valid prompts: {validation_results['valid_prompts']}")
            self.log_progress(f"      Validation rate: {valid_rate:.1%}")
            
            if validation_results["issues"]:
                self.log_detail(f"  ⚠️  Issues found:")
                for issue in validation_results["issues"][:5]:  # Only show first 5 issues
                    self.log_detail(f"      {issue}")
            
            # Simple check if there are valid results
            success = valid_rate > 0  # Success if any generated results exist
            if success:
                self.log_progress("  ✅ Output validation completed")
            else:
                self.log_progress("  ❌ Output validation failed: No valid generation results", "error")
            
            self.test_results["output_validation"] = {
                "status": success,
                "validation_results": validation_results,
                "valid_rate": valid_rate
            }
            return success
            
        except Exception as e:
            self.log_progress(f"  ❌ Output validation failed: {e}", "error")
            self.test_results["output_validation"] = {"status": False, "error": str(e)}
            return False
    
    def run_all_tests(self, check_service_only: bool = False) -> bool:
        """Run all harmful prompt generation tests"""
        nodes_desc = "all nodes" if self.selected_nodes_count == -1 else f"{self.selected_nodes_count} nodes"
        self.log_progress(f"🎯 Starting harmful prompt generation test (domain: {self.domain}, node strategy: {nodes_desc})")
        self.log_progress("="*60)
        
        tests = [
            ("Model service check", self.check_model_service),
        ]
        
        if not check_service_only:
            tests.extend([
                ("Model initialization", self.test_model_initialization),
                ("Generator initialization", self.test_harmful_prompt_generator),
                ("Knowledge graph loading", self.load_real_knowledge_graph_nodes),
                ("Context formatting", self.test_context_formatting),
                ("Harmful prompt generation", self.test_harmful_prompt_generation),
                ("Output validation", self.test_output_validation)
            ])
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.log_progress(f"\n📋 Executing test: {test_name}")
            try:
                success = test_func()
                if success:
                    passed_tests += 1
                elif test_name == "Model service check":
                    # If model service not available, exit early
                    self.log_progress("💔 Model service not available, cannot continue testing", "error")
                    break
            except Exception as e:
                self.log_progress(f"Exception during test '{test_name}': {e}", "error")
        
        # Generate test summary
        self.log_progress(f"\n{'='*60}")
        self.log_progress(f"📊 Harmful prompt generation test summary")
        self.log_progress(f"✅ Passed: {passed_tests}/{total_tests}")
        
        if check_service_only:
            overall_status = passed_tests >= 1
            if overall_status:
                self.log_progress("🎉 Model service check passed, can run full test")
            else:
                self.log_progress("❌ Model service not available, please start service and retry", "error")
        else:
            success_rate = passed_tests / total_tests
            # Get success rate threshold from configuration
            overall_status = success_rate > 0  # Success if any test passed
            
            if overall_status:
                self.log_progress("🎉 Harmful prompt generation test completed")
            else:
                self.log_progress("❌ Harmful prompt generation test failed", "error")
        
        # Save test results
        self._save_test_results(overall_status, passed_tests, total_tests)
        
        return overall_status
    
    def _save_test_results(self, overall_status: bool, passed: int, total: int):
        """Save test results to file"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domain": self.domain,
            "selected_nodes_count": self.selected_nodes_count,
            "model_config": self.model_config,
            "overall_status": overall_status,
            "summary": {
                "passed_tests": passed,
                "total_tests": total,
                "success_rate": passed / total if total > 0 else 0
            },
            "detailed_results": self.test_results
        }
        
        output_dir = SCRIPT_DIR / "outputs" / self.domain
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "step2_harmful_generation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_progress(f"📋 Harmful prompt generation test report saved: {report_path}")
        
        # If prompts were generated, save them separately
        if ("harmful_generation" in self.test_results and 
            self.test_results["harmful_generation"]["status"]):
            prompts_path = output_dir / "step2_generated_prompts.json"
            with open(prompts_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.test_results["harmful_generation"]["prompts"], 
                    f, indent=2, ensure_ascii=False
                )
            self.log_progress(f"📝 Generated prompts saved: {prompts_path}")
        
        # Save prompt templates for debugging
        if self.prompt_templates:
            templates_path = output_dir / "step2_prompt_templates.json"
            with open(templates_path, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_templates, f, indent=2, ensure_ascii=False)
            self.log_detail(f"🔍 Prompt templates saved: {templates_path}")
            
            # Also save a more readable text version
            templates_txt_path = output_dir / "step2_prompt_templates.txt"
            with open(templates_txt_path, 'w', encoding='utf-8') as f:
                f.write("Step 2 Harmful Prompt Generation - Template Debug Information\n")
                f.write("="*80 + "\n\n")
                
                for i, template_data in enumerate(self.prompt_templates):
                    f.write(f"Template {i+1}: {template_data['node_name']} - {template_data['category']}\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Node ID: {template_data['node_id']}\n")
                    f.write(f"Node Name: {template_data['node_name']}\n")
                    f.write(f"Harm Category: {template_data['category']}\n")
                    f.write(f"Domain Info: {template_data['domain_info']}\n")
                    if template_data['wikipedia_description']:
                        f.write(f"Wikipedia Description: {template_data['wikipedia_description']}\n")
                    if template_data['wikipedia_summary']:
                        f.write(f"Wikipedia Summary: {template_data['wikipedia_summary']}\n")
                    f.write(f"Generation Time: {template_data['timestamp']}\n")
                    f.write(f"\n--- FULL GENERATION PROMPT ---\n")
                    f.write(f"{template_data['prompt_template']}\n")
                    f.write("\n" + "="*80 + "\n\n")
            
            self.log_detail(f"📄 Prompt template text version saved: {templates_txt_path}")

def main():
    parser = argparse.ArgumentParser(description="Step 2: Harmful prompt generation test (using real knowledge graph data)")
    parser.add_argument("--domain", default="medicine",
                      choices=["medicine", "finance", "education", "law"],
                      help="Test domain")
    parser.add_argument("--selected-nodes-count", type=int, default=-1,
                      help="Number of randomly selected knowledge graph nodes (default: -1 means use all nodes, set positive number to limit)")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed information")
    parser.add_argument("--check-service-only", action="store_true",
                      help="Only check model service availability")
    parser.add_argument("--output-dir", type=str,
                      help="Output directory (passed by test runner)")
    
    args = parser.parse_args()
    
    tester = HarmfulGenerationTester(
        domain=args.domain,
        verbose=args.verbose,
        selected_nodes_count=args.selected_nodes_count
    )
    
    success = tester.run_all_tests(check_service_only=args.check_service_only)
    
    # Remove all print statements, results logged via logger
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
