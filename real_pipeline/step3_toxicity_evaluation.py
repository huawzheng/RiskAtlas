#!/usr/bin/env python
"""
Step 3: Toxicity Evaluation Test
================================

Test using Granite Guardian to evaluate toxicity of harmful prompts:
- Check Granite Guardian service availability
- Load previously generated harmful prompts
- Call Granite Guardian for toxicity evaluation
- Validate evaluation result format and reasonableness
- Save evaluated data

Note: This step requires switching to Granite Guardian server (port 8001)
      Please stop step3's fine-tuned model first, then start Granite Guardian

Usage:
    python step3_toxicity_evaluation.py --domain medicine
    python step3_toxicity_evaluation.py --domain finance --test-size 10
    python step3_toxicity_evaluation.py --domain medicine --check-service-only

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

from src.utils.logger_utils import get_logger
from src.modules.granite_guardian_evaluator import GraniteGuardianEvaluator

class ToxicityEvaluationTester:
    """Toxicity evaluation tester"""
    
    def __init__(self, domain: str = "medicine", test_size: int = None, verbose: bool = False):
        self.domain = domain
        self.test_size = test_size
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        
        # Load domain configuration
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            self.domain_config = config_manager.load_domain_config(domain)
        except Exception as e:
            self.logger.warning(f"Unable to load domain configuration: {e}")
            self.domain_config = None
        
        # Granite Guardian service configuration
        self.model_config = {
            "base_url": "http://localhost:8001/v1",
            "model_name": "granite-guardian-3.1-8b-vllm-server",
        }

        
        self.test_results = {}
        self.granite_evaluator = None
    
    def check_granite_service(self) -> bool:
        """Check Granite Guardian service availability"""
        self.logger.info("🛡️  Checking Granite Guardian service...")
        
        try:
            # Get timeout from configuration
            timeout = self.domain_config.get('test_params', {}).get('request_timeout', 10) if self.domain_config else 10
            
            # Check service health status
            health_url = f"{self.model_config['base_url']}/models"
            response = requests.get(health_url, timeout=timeout)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                
                if self.model_config["model_name"] in available_models:
                    self.logger.info(f"  ✅ Granite Guardian service available: {self.model_config['model_name']}")
                    self.logger.info(f"  📊 Available models: {available_models}")
                    
                    self.test_results["granite_service"] = {
                        "status": True,
                        "available_models": available_models,
                        "target_model": self.model_config["model_name"]
                    }
                    return True
                else:
                    self.logger.error(f"  ❌ Granite Guardian model not available: {self.model_config['model_name']}")
                    self.logger.error(f"  📊 Available models: {available_models}")
                    return False
            else:
                self.logger.error(f"  ❌ Granite Guardian service not available: HTTP {response.status_code}")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"  ❌ Granite Guardian service connection failed: {e}")
            self.logger.error(f"  💡 Please make sure to run: bash scripts/start_vllm_granite_guardian_server.sh")
            self.test_results["granite_service"] = {"status": False, "error": str(e)}
            return False
    
    def test_granite_initialization(self) -> bool:
        """Test Granite Guardian initialization"""
        self.logger.info("🔧 Testing Granite Guardian initialization...")
        
        try:
            # Initialize LLM manager
            from src.models.llm_factory import LLMManager
            llm_manager = LLMManager()
            
            # Initialize Granite Guardian evaluator
            self.granite_evaluator = GraniteGuardianEvaluator(llm_manager=llm_manager)
            
            # Test simple safety evaluation
            test_prompt = "Tell me about the weather today."
            result = self.granite_evaluator.evaluate_prompt(test_prompt)
            
            if result and hasattr(result, 'is_harmful'):
                self.logger.info("  ✅ Granite Guardian initialization successful")
                if self.verbose:
                    self.logger.info(f"  📊 Test evaluation result: harmful={result.is_harmful}, toxicity_score={result.toxicity_score:.3f}")
                
                self.test_results["granite_initialization"] = {
                    "status": True,
                    "test_result": {
                        "is_harmful": result.is_harmful,
                        "toxicity_score": result.toxicity_score
                    }
                }
                return True
            else:
                self.logger.error("  ❌ Granite Guardian evaluation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"  ❌ Granite Guardian initialization failed: {e}")
            self.test_results["granite_initialization"] = {"status": False, "error": str(e)}
            return False
    
    def load_generated_prompts(self) -> List[Dict]:
        """Load previously generated harmful prompts"""
        self.logger.info("📂 Loading generated harmful prompts...")
        
        # Try to load from step2 output
        prompts_path = SCRIPT_DIR / "outputs" / self.domain / "step2_generated_prompts.json"
        
        if prompts_path.exists():
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                self.logger.info(f"  ✅ Loaded {len(prompts)} prompts from step2")
                
                # Convert data format for evaluator, preserve complete node info
                formatted_prompts = []
                for prompt_data in prompts:
                    formatted_prompt = {
                        "prompt_id": prompt_data.get("prompt_id", f"unknown_{len(formatted_prompts)}"),
                        "prompt": prompt_data.get("prompt_text", ""),
                        "entity": prompt_data.get("node_name", "unknown"),
                        "category": prompt_data.get("category", "unknown"),
                        "behavior_type": prompt_data.get("behavior_type", "unknown"),  # New: behavior type
                        "node_id": prompt_data.get("node_id", "unknown"),
                        "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),  # Modified: use wikidata_uri
                        "timestamp": prompt_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                        # Preserve detailed node information
                        "node_info": {
                            "id": prompt_data.get("node_id", "unknown"),
                            "name": prompt_data.get("node_name", "unknown"),
                            "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                            "description": prompt_data.get("wikipedia_description", ""),
                            "domain": prompt_data.get("domain", "unknown")
                        }
                    }
                    formatted_prompts.append(formatted_prompt)
                
                return formatted_prompts if self.test_size is None else formatted_prompts[:self.test_size]  # If test_size not specified, use all data
                
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to load step2 prompts: {e}")
        
        # If no available prompts, return empty list
        self.logger.error(f"  ❌ Prompt file not found: {prompts_path}")
        self.logger.error(f"  💡 Please run step2 first to generate harmful prompts")
        return []
    
    def test_toxicity_evaluation(self) -> bool:
        """Test toxicity evaluation"""
        self.logger.info("🔍 Testing toxicity evaluation...")
        
        if not self.granite_evaluator:
            self.logger.error("  ❌ Granite Guardian evaluator not initialized")
            return False
        
        # Load prompts to evaluate
        prompts_to_evaluate = self.load_generated_prompts()
        
        if not prompts_to_evaluate:
            self.logger.error("  ❌ No prompts to evaluate")
            return False
        
        try:
            evaluated_prompts = []
            
            for i, prompt_data in enumerate(prompts_to_evaluate):
                prompt_text = prompt_data["prompt"]
                prompt_id = prompt_data.get("prompt_id", f"prompt_{i+1}")
                
                self.logger.info(f"  🔍 Evaluating prompt {i+1}/{len(prompts_to_evaluate)}: {prompt_id}")
                
                try:
                    # Perform toxicity evaluation
                    evaluation_result = self.granite_evaluator.evaluate_prompt(prompt_text)
                    
                    if evaluation_result:
                        evaluated_prompt = {
                            **prompt_data,  # Preserve original data
                            "toxicity_evaluation": {
                                "is_harmful": evaluation_result.is_harmful,
                                "toxicity_score": evaluation_result.toxicity_score,
                                "reasoning": evaluation_result.reasoning,
                                "raw_response": evaluation_result.raw_response
                            },
                            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        evaluated_prompts.append(evaluated_prompt)
                        
                        if self.verbose:
                            is_harmful = evaluation_result.is_harmful
                            toxicity_score = evaluation_result.toxicity_score
                            self.logger.info(f"    📊 Harmful: {is_harmful}, Toxicity score: {toxicity_score:.3f}")
                    
                    time.sleep(1)  # Avoid too frequent requests
                    
                except Exception as e:
                    self.logger.warning(f"    ⚠️  Prompt {prompt_id} evaluation failed: {e}")
                    # Even if evaluation fails, record original prompt
                    evaluated_prompts.append({
                        **prompt_data,
                        "toxicity_evaluation": {"error": str(e)},
                        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue
            
            if evaluated_prompts:
                self.logger.info(f"  ✅ Successfully evaluated {len(evaluated_prompts)} prompts")
                self.test_results["toxicity_evaluation"] = {
                    "status": True,
                    "evaluated_count": len(evaluated_prompts),
                    "prompts": evaluated_prompts
                }
                return True
            else:
                self.logger.error("  ❌ No prompts successfully evaluated")
                self.test_results["toxicity_evaluation"] = {"status": False, "evaluated_count": 0}
                return False
                
        except Exception as e:
            self.logger.error(f"  ❌ Toxicity evaluation test failed: {e}")
            self.test_results["toxicity_evaluation"] = {"status": False, "error": str(e)}
            return False
    
    def test_evaluation_analysis(self) -> bool:
        """Test evaluation result analysis"""
        self.logger.info("📊 Testing evaluation result analysis...")
        
        if ("toxicity_evaluation" not in self.test_results or 
            not self.test_results["toxicity_evaluation"]["status"]):
            self.logger.error("  ❌ No evaluation results to analyze")
            return False
        
        try:
            evaluated_prompts = self.test_results["toxicity_evaluation"]["prompts"]
            analysis = {
                "total_prompts": len(evaluated_prompts),
                "safe_prompts": 0,
                "unsafe_prompts": 0,
                "evaluation_errors": 0
            }
            
            for prompt_data in evaluated_prompts:
                eval_result = prompt_data.get("toxicity_evaluation", {})
                
                if "error" in eval_result:
                    analysis["evaluation_errors"] += 1
                else:
                    is_harmful = eval_result.get("is_harmful")
                    if is_harmful is True:
                        analysis["unsafe_prompts"] += 1
                    elif is_harmful is False:
                        analysis["safe_prompts"] += 1
            
            # Output analysis results
            self.logger.info(f"  📊 Evaluation analysis results:")
            self.logger.info(f"      Total prompts: {analysis['total_prompts']}")
            self.logger.info(f"      Safe prompts: {analysis['safe_prompts']}")
            self.logger.info(f"      Unsafe prompts: {analysis['unsafe_prompts']}")
            self.logger.info(f"      Evaluation errors: {analysis['evaluation_errors']}")
            
            # Check if analysis completed successfully
            success_rate = (analysis["safe_prompts"] + analysis["unsafe_prompts"]) / analysis["total_prompts"]
            success = success_rate > 0  # Success if any evaluation results exist
            
            if success:
                self.logger.info("  ✅ Evaluation result analysis completed")
            else:
                self.logger.error("  ❌ Evaluation result analysis failed: No valid evaluation results")
            
            self.test_results["evaluation_analysis"] = {
                "status": success,
                "analysis": analysis,
                "success_rate": success_rate
            }
            return success
            
        except Exception as e:
            self.logger.error(f"  ❌ Evaluation result analysis failed: {e}")
            self.test_results["evaluation_analysis"] = {"status": False, "error": str(e)}
            return False
    
    def run_all_tests(self, check_service_only: bool = False) -> bool:
        """Run all toxicity evaluation tests"""
        test_size_info = f"all data" if self.test_size is None else str(self.test_size)
        self.logger.info(f"\n🛡️  Starting toxicity evaluation tests (domain: {self.domain}, size: {test_size_info})")
        self.logger.info("="*60)
        
        tests = [
            ("Granite Guardian Service Check", self.check_granite_service),
        ]
        
        if not check_service_only:
            tests.extend([
                ("Granite Guardian Initialization", self.test_granite_initialization),
                ("Toxicity Evaluation", self.test_toxicity_evaluation),
                ("Evaluation Result Analysis", self.test_evaluation_analysis)
            ])
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n📋 Executing test: {test_name}")
            try:
                success = test_func()
                if success:
                    passed_tests += 1
                elif test_name == "Granite Guardian Service Check":
                    # If Granite Guardian service is unavailable, exit early
                    self.logger.error("💔 Granite Guardian service unavailable, cannot continue tests")
                    break
            except Exception as e:
                self.logger.error(f"Exception occurred during test '{test_name}': {e}")
        
        # Generate test summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Toxicity Evaluation Test Summary")
        self.logger.info(f"✅ Passed: {passed_tests}/{total_tests}")
        
        if check_service_only:
            overall_status = passed_tests >= 1
            if overall_status:
                self.logger.info("🎉 Granite Guardian service check passed, ready to run full tests")
            else:
                self.logger.error("❌ Granite Guardian service unavailable, please start the service and retry")
        else:
            overall_status = passed_tests > 0
            
            if overall_status:
                self.logger.info("🎉 Toxicity evaluation tests completed")
            else:
                self.logger.error("❌ Toxicity evaluation tests failed")
        
        # Save test results
        self._save_test_results(overall_status, passed_tests, total_tests)
        
        return overall_status
    
    def _save_test_results(self, overall_status: bool, passed: int, total: int):
        """Save test results to file"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domain": self.domain,
            "test_size": self.test_size,
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
        
        report_path = output_dir / "step3_toxicity_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Toxicity evaluation test report saved: {report_path}")
        
        # If there are evaluated prompts, save separately
        if ("toxicity_evaluation" in self.test_results and 
            self.test_results["toxicity_evaluation"]["status"]):
            evaluated_path = output_dir / "step3_evaluated_prompts.json"
            with open(evaluated_path, 'w', encoding='utf-8') as f:
                json.dump(
                    self.test_results["toxicity_evaluation"]["prompts"], 
                    f, indent=2, ensure_ascii=False
                )
            self.logger.info(f"📝 Evaluated prompts saved: {evaluated_path}")

def main():
    parser = argparse.ArgumentParser(description="Step 3: Toxicity Evaluation Test")
    parser.add_argument("--domain", default="medicine",
                      choices=["medicine", "finance", "education", "law"],
                      help="Test domain")
    parser.add_argument("--test-size", type=int, default=None,
                      help="Test data size (default: all data)")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed information")
    parser.add_argument("--check-service-only", action="store_true",
                      help="Only check Granite Guardian service availability")
    parser.add_argument("--output-dir", type=str,
                      help="Output directory (passed by test runner)")
    
    args = parser.parse_args()
    
    tester = ToxicityEvaluationTester(
        domain=args.domain,
        test_size=args.test_size,
        verbose=args.verbose
    )
    
    success = tester.run_all_tests(check_service_only=args.check_service_only)
    
    if success:
        if args.check_service_only:
            print(f"\n✅ Step 3 Service Check: Granite Guardian service available")
        else:
            print(f"\n✅ Step 3 Complete: Toxicity evaluation tests passed")
    else:
        if args.check_service_only:
            print(f"\n❌ Step 3 Service Check: Granite Guardian service unavailable")
        else:
            print(f"\n❌ Step 3 Failed: Toxicity evaluation tests failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
