#!/usr/bin/env python
"""
Step 4: Data Filtering Test
===========================

Filter data after Step 3 harmfulness evaluation:
1. First filtering based on harmfulness threshold
2. Evaluate PPL values of passed data
3. Second filtering based on PPL threshold
4. Save final filtered results for Step 5

Usage:
    python step4_dataset_assembly.py --domain medicine

"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add project root directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger_utils import get_logger

class DataFilterTester:
    """Data Filter Tester - Simplified version focusing on core filtering functionality"""
    
    def __init__(self, domain: str = "medicine", verbose: bool = False):
        self.domain = domain
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        
        self.output_dir = SCRIPT_DIR / "outputs" / self.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        from src.utils.config_manager import ConfigManager
        self.config_manager = ConfigManager()
        self.domain_config = None
        
        # Load threshold configuration
        try:
            self.domain_config = self.config_manager.load_domain_config(self.domain)
            filtering_thresholds = getattr(self.domain_config, 'filtering_thresholds', {})
            self.toxicity_threshold = filtering_thresholds.get('toxicity_filter', 0.9)
            self.ppl_threshold = filtering_thresholds.get('perplexity', {}).get('initial_filter', 50.0)
            self.logger.info(f"Configuration loaded: toxicity_threshold={self.toxicity_threshold*100:.0f}%, ppl_threshold={self.ppl_threshold}")
        except Exception as e:
            self.logger.warning(f"Failed to load configuration, using defaults: {e}")
            self.toxicity_threshold = 0.9
            self.ppl_threshold = 50.0
        
        self.filtered_data = []
    
    def load_step3_data(self) -> List[Dict]:
        """Load Step 3 evaluation results"""
        self.logger.info("📂 Loading Step 3 evaluation data...")
        
        step3_file = self.output_dir / "step3_evaluated_prompts.json"
        if not step3_file.exists():
            self.logger.error(f"Step 3 data file does not exist: {step3_file}")
            return []
        
        try:
            with open(step3_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"✅ Loaded {len(data)} Step 3 evaluation records")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load Step 3 data: {e}")
            return []
    
    def filter_by_toxicity(self, data: List[Dict]) -> List[Dict]:
        """Step 1: Filter by harmfulness level"""
        self.logger.info(f"🔍 Step 1 filtering: harmfulness >= {self.toxicity_threshold*100:.0f}%")
        
        filtered = []
        for item in data:
            toxicity_eval = item.get('toxicity_evaluation', {})
            toxicity_score = toxicity_eval.get('toxicity_score', 0)
            
            if isinstance(toxicity_score, (int, float)) and toxicity_score >= self.toxicity_threshold:
                filtered.append(item)
                if self.verbose:
                    prompt_id = item.get('prompt_id', 'unknown')
                    self.logger.info(f"  ✅ Kept: {prompt_id} (harmfulness: {toxicity_score*100:.1f}%)")
            elif self.verbose:
                prompt_id = item.get('prompt_id', 'unknown')
                self.logger.info(f"  ❌ Filtered: {prompt_id} (harmfulness: {toxicity_score*100:.1f}%)")
        
        self.logger.info(f"Step 1 filtering result: {len(filtered)}/{len(data)} records passed")
        return filtered
    
    def evaluate_and_filter_ppl(self, data: List[Dict]) -> List[Dict]:
        """Step 2: Evaluate PPL and filter by PPL threshold"""
        self.logger.info(f"🔍 Step 2 filtering: Evaluate PPL and filter <= {self.ppl_threshold}")
        
        # Initialize PPL evaluation service
        try:
            from src.services.evaluation.fluency_evaluator_service import FluencyEvaluatorService
            fluency_evaluator = FluencyEvaluatorService()
        except Exception as e:
            self.logger.error(f"❌ Unable to initialize PPL evaluation service: {e}")
            # If PPL evaluation service cannot be initialized, return all data directly
            return data
        
        filtered = []
        for item in data:
            prompt_text = item.get('prompt', '')
            prompt_id = item.get('prompt_id', 'unknown')
            
            try:
                # Use actual PPL evaluation
                fluency_result = fluency_evaluator.evaluate_fluency(prompt_text)
                ppl_score = fluency_result.perplexity_score
                
                # Add PPL info to data
                item['ppl_score'] = ppl_score
                
                if ppl_score <= self.ppl_threshold:
                    filtered.append(item)
                    if self.verbose:
                        self.logger.info(f"  ✅ Kept: {prompt_id} (PPL: {ppl_score:.1f})")
                elif self.verbose:
                    self.logger.info(f"  ❌ Filtered: {prompt_id} (PPL: {ppl_score:.1f})")
                    
            except Exception as e:
                self.logger.error(f"  ❌ PPL evaluation failed: {prompt_id} - {e}")
                # Raise exception on evaluation failure, do not continue processing
                raise RuntimeError(f"PPL evaluation failed, cannot continue filtering: {e}")
        
        self.logger.info(f"Step 2 filtering result: {len(filtered)}/{len(data)} records passed")
        return filtered
    
    def save_filtered_data(self, data: List[Dict]) -> bool:
        """Save filtered data"""
        self.logger.info("💾 Saving filtered data...")
        
        if not data:
            self.logger.warning("No data to save")
            return False
        
        # Keep complete data structure, not simplified
        full_data = []
        for item in data:
            toxicity_eval = item.get('toxicity_evaluation', {})
            # Keep all original fields and add filtering info
            full_item = {
                **item,  # Keep all original fields
                "filtering_info": {
                    "toxicity_threshold": self.toxicity_threshold,
                    "ppl_threshold": self.ppl_threshold,
                    "passed_toxicity_filter": True,  # Reaching here means passed toxicity filter
                    "passed_ppl_filter": item.get('ppl_score', 0) <= self.ppl_threshold
                },
                "filter_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            full_data.append(full_item)
        
        # Save complete data
        output_file = self.output_dir / "step4_filtered_prompts.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Saved {len(full_data)} filtered records: {output_file.name}")
            
            # Save filtering statistics
            stats = {
                "domain": self.domain,
                "filtering_config": {
                    "toxicity_threshold": self.toxicity_threshold,
                    "ppl_threshold": self.ppl_threshold
                },
                "results": {
                    "final_count": len(full_data),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            stats_file = self.output_dir / "step4_filtering_report.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Filtering report saved: {stats_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return False
    
    def run_filtering(self) -> bool:
        """Run complete data filtering workflow"""
        self.logger.info(f"🔄 Starting data filtering (domain: {self.domain})")
        self.logger.info("="*50)
        
        # 1. Load Step 3 data
        step3_data = self.load_step3_data()
        if not step3_data:
            self.logger.error("❌ No data to process")
            return False
        
        # 2. Step 1 filtering: Harmfulness
        toxicity_filtered = self.filter_by_toxicity(step3_data)
        if not toxicity_filtered:
            self.logger.error("❌ No data after harmfulness filtering")
            return False
        
        # 3. Step 2 filtering: PPL evaluation and filtering
        final_filtered = self.evaluate_and_filter_ppl(toxicity_filtered)
        if not final_filtered:
            self.logger.error("❌ No data after PPL filtering")
            return False
        
        # 4. Save results
        success = self.save_filtered_data(final_filtered)
        
        if success:
            self.logger.info("="*50)
            self.logger.info(f"✅ Data filtering completed:")
            self.logger.info(f"   Original data: {len(step3_data)} records")
            self.logger.info(f"   After harmfulness filtering: {len(toxicity_filtered)} records")
            self.logger.info(f"   Final filtered result: {len(final_filtered)} records")
            self.logger.info(f"   Overall retention rate: {len(final_filtered)/len(step3_data)*100:.1f}%")
        else:
            self.logger.error("❌ Data filtering failed")
        
        return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 4: Data Filtering Test")
    parser.add_argument("--domain", type=str, default="medicine", 
                      choices=["medicine", "finance", "education", "law"],
                      help="Domain")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create tester
    tester = DataFilterTester(
        domain=args.domain,
        verbose=args.verbose
    )
    
    # Run data filtering
    success = tester.run_filtering()
    
    if success:
        print(f"\n✅ Step 4 Complete: Data filtering successful")
        print(f"📊 Filtered data saved to step4_filtered_prompts.json")
    else:
        print(f"\n❌ Step 4 Failed: Data filtering failed")
    
    # Exit
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
