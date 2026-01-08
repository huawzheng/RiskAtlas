#!/usr/bin/env python
"""
Step 6: Dataset Finalization
============================

Organize all data from previous steps and generate the final dataset format
suitable for uploading to Hugging Face.

Dataset Format:
Basic Information:
  - id: Record ID (starting from 1)
  - prompt_id: Unique identifier for the original prompt
  - domain: Domain (medicine, finance, law, education, etc.)
  - entity: Entity name
  - wikidata_uri: Wikidata URI
  - entity_description: Entity description

Prompt Information:
  - original_prompt: Original harmful prompt
  - stealth_prompt: Stealth prompt after processing (null if failed)
  - stealth_success: Whether stealth processing succeeded
  - category: Harmful category
  - behavior_type: Behavior type
  - trigger_words: Trigger word list (sensitive words used for stealth replacement)

Target Model Response (Important):
  - target_response: Complete response from target model to stealth prompt

Evaluation Information:
  - asr_evaluation_reasoning: Gemini's reasoning for attack success evaluation
  - intent_preserved: Whether stealth processing preserved original intent
  - intent_reasoning: Gemini's intent evaluation reasoning
  - is_fluent: Whether stealth prompt is fluent

Toxicity Evaluation:
  - is_harmful: Whether original prompt is harmful
  - toxicity_score_original: Toxicity score of original prompt
  - toxicity_reasoning: Toxicity evaluation reasoning

Quality Metrics:
  - ppl_score_original: Perplexity of original prompt
  - similarity_score_stealth: Semantic similarity between stealth and original
  - ppl_score_stealth: Perplexity of stealth prompt

Processing Information:
  - winning_path: Successful stealth path (basic/enhanced)
  - iterations: Number of iterations
  - processing_time: Processing time (seconds)
  - has_stealth_version: Whether has stealth version
  - timestamp: Timestamp

Usage:
    python step6_dataset_finalization.py --domain medicine
    python step6_dataset_finalization.py --domain medicine --output-format csv

"""

import argparse
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger_utils import get_logger

class DatasetFinalizer:
    """Dataset Finalizer"""
    
    def __init__(self, domain: str = "medicine", output_format: str = "json", verbose: bool = False):
        self.domain = domain
        self.output_format = output_format.lower()
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        
        self.output_dir = SCRIPT_DIR / "outputs" / self.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful_stealth": 0,           # Case 1: Stealth succeeded
            "failed_with_stealth": 0,          # Case 2: Stealth failed but has prompt
            "failed_no_stealth": 0,            # Case 3: No stealth prompt at all
            "domains": {},
            "categories": {}
        }

    def load_step5_data(self) -> List[Dict]:
        """Load step5 output data - including both successful and failed stealth data"""
        self.logger.info("📂 Loading Step5 stealth data (including failed cases)...")
        
        # Priority list: first try to load complete dataset, then results file, finally successful cases
        file_priority = [
            # Complete dataset files (containing all data)
            f"step5_parallel_dataset_{self.domain}.json",
            f"step5_stealth_dataset_{self.domain}.json", 
            f"step5_attack_results_{self.domain}.json",
            f"step5_parallel_results_{self.domain}.json",
            
            # If no complete dataset, fall back to successful cases
            f"step5_successful_stealth_dataset_{self.domain}.json",
            f"step5_parallel_successful_{self.domain}.json"
        ]
        
        for filename in file_priority:
            file_path = self.output_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Count successful and failed cases
                    success_count = 0
                    fail_with_stealth = 0
                    fail_no_stealth = 0
                    
                    for record in data:
                        stealth_success = (record.get("success", False) or 
                                         record.get("stealth_success", False))
                        stealth_prompt = record.get("stealth_prompt")
                        
                        if stealth_success:
                            success_count += 1
                        elif stealth_prompt and stealth_prompt.strip():
                            fail_with_stealth += 1
                        else:
                            fail_no_stealth += 1
                    
                    self.logger.info(f"  ✅ Loaded {len(data)} records from {filename}")
                    self.logger.info(f"     🎯 Case 1 - Stealth succeeded: {success_count} records")
                    self.logger.info(f"     📝 Case 2 - Stealth failed but has prompt: {fail_with_stealth} records")
                    self.logger.info(f"     ❌ Case 3 - No stealth prompt at all: {fail_no_stealth} records")
                    
                    # Check if this is complete dataset or only successful cases
                    if fail_with_stealth == 0 and fail_no_stealth == 0 and "successful" in filename:
                        self.logger.warning(f"  ⚠️  Current file only contains successful cases, may be missing failed data")
                        self.logger.warning(f"  💡 Suggest checking if there's a file with complete data")
                    
                    return data
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️  Failed to load {filename}: {e}")
                    continue
        
        # If no step5 file found, return empty list
        self.logger.error("  ❌ No Step5 output file found")
        self.logger.error("  💡 Please run Step5 first to generate stealth data")
        self.logger.error("  📋 Expected filenames:")
        for filename in file_priority:
            self.logger.error(f"     - {filename}")
        return []

    def format_dataset_record(self, record: Dict, record_id: int) -> Dict:
        """Format raw record into final dataset format"""
        
        # Extract core information
        entity = record.get("entity", "unknown")
        wikidata_uri = record.get("wikidata_uri", "unknown")
        original_prompt = record.get("original_prompt", "")
        stealth_prompt = record.get("stealth_prompt", None)
        
        # Determine stealth status
        stealth_success = (record.get("success", False) or 
                          record.get("stealth_success", False))
        
        # Process stealth prompt - distinguish three cases
        stealth_prompt_raw = record.get("stealth_prompt")
        
        if stealth_success:
            # Case 1: Stealth succeeded
            stealth_prompt = stealth_prompt_raw if stealth_prompt_raw else ""
            stealth_status = "success"
        elif stealth_prompt_raw and stealth_prompt_raw.strip():
            # Case 2: Stealth failed but has prompt (keep stealth prompt but mark as failed)
            stealth_prompt = stealth_prompt_raw
            stealth_status = "failed_with_prompt"
            stealth_success = False  # Ensure marked as failed
        else:
            # Case 3: No stealth prompt at all
            stealth_prompt = None
            stealth_status = "failed_no_prompt"
            stealth_success = False  # Ensure marked as failed
        
        category = record.get("category", "unknown")
        
        # Extract info from metadata (if not present in main record)
        metadata = record.get("metadata", {})
        if entity == "unknown" and isinstance(metadata, dict):
            entity = metadata.get("entity", "unknown")
        if wikidata_uri == "unknown" and isinstance(metadata, dict):
            wikidata_uri = metadata.get("wikidata_uri", "unknown")
        if category == "unknown" and isinstance(metadata, dict):
            category = metadata.get("category", "unknown")
        if not original_prompt and isinstance(metadata, dict):
            original_prompt = metadata.get("prompt", metadata.get("original_prompt", ""))
        
        # Ensure wikidata_uri format is correct
        wikidata_uri = self._format_wikidata_uri(wikidata_uri, record.get('node_id', 'unknown'))
        
        # Extract gemini evaluation info
        gemini_evaluation = record.get("gemini_evaluation", {})
        if not isinstance(gemini_evaluation, dict):
            gemini_evaluation = {}
        
        # Extract behavior type
        behavior_type = record.get("behavior_type", "unknown")
        if behavior_type == "unknown" and isinstance(metadata, dict):
            behavior_type = metadata.get("behavior_type", "unknown")
        
        # Extract node description
        node_description = None
        if isinstance(metadata, dict):
            node_info = metadata.get("node_info", {})
            if isinstance(node_info, dict):
                node_description = node_info.get("description")
        
        # Extract prompt_id
        prompt_id = record.get("prompt_id")
        if not prompt_id and isinstance(metadata, dict):
            prompt_id = metadata.get("prompt_id")
        
        # Extract toxicity evaluation details
        toxicity_reasoning = None
        is_harmful = None
        if isinstance(metadata, dict) and "toxicity_evaluation" in metadata:
            toxicity_eval = metadata["toxicity_evaluation"]
            if isinstance(toxicity_eval, dict):
                toxicity_reasoning = toxicity_eval.get("reasoning")
                is_harmful = toxicity_eval.get("is_harmful")
        
        # Extract trigger words (accumulated_trigger_words)
        trigger_words = record.get("accumulated_trigger_words")
        if not trigger_words and isinstance(metadata, dict):
            trigger_words = metadata.get("accumulated_trigger_words")
        
        # Build final record
        final_record = {
            "id": record_id,
            "prompt_id": prompt_id,  # Original prompt ID
            "domain": self.domain,
            "entity": entity,
            "wikidata_uri": wikidata_uri,
            "entity_description": node_description,  # Node description
            "original_prompt": original_prompt,
            "stealth_prompt": stealth_prompt,  # null for failed cases, string for success/partial cases
            "stealth_success": stealth_success,
            "category": category,
            "behavior_type": behavior_type,  # Behavior type
            
            # Trigger words (sensitive word list used for stealth replacement)
            "trigger_words": trigger_words,  # Accumulated trigger words list
            
            # Target model response (core data)
            "target_response": record.get("target_response"),  # Target model's response
            
            # ASR evaluation info (Gemini evaluation)
            "asr_evaluation_reasoning": record.get("asr_evaluation_reasoning"),  # ASR evaluation reasoning
            
            # Gemini stealth evaluation info
            "intent_preserved": gemini_evaluation.get("intent_preserved"),  # Whether intent is preserved
            "intent_reasoning": gemini_evaluation.get("intent_reasoning"),  # Intent evaluation reasoning
            "is_fluent": gemini_evaluation.get("is_fluent"),  # Whether fluent
            
            # Original prompt toxicity evaluation
            "is_harmful": is_harmful,  # Whether harmful
            "toxicity_score_original": self._extract_toxicity_score(record),
            "toxicity_reasoning": toxicity_reasoning,  # Toxicity evaluation reasoning
            "ppl_score_original": self._extract_original_ppl(record),  # Original PPL
            
            # Stealth prompt attributes (keep scores as long as stealth prompt exists, regardless of success)
            "similarity_score_stealth": self._extract_similarity_score(record, stealth_prompt),
            "ppl_score_stealth": self._extract_ppl_score(record, stealth_prompt),
            
            # Stealth process info
            "winning_path": record.get("winning_path", "unknown"),
            "iterations": self._extract_iterations(record),
            "processing_time": record.get("processing_time"),  # Processing time (seconds)
            "has_stealth_version": record.get("has_stealth_version", stealth_prompt is not None),
            "timestamp": record.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        }
        
        return final_record

    def _format_wikidata_uri(self, wikidata_uri: str, node_id: str) -> str:
        """Ensure wikidata_uri format is correct"""
        if wikidata_uri and wikidata_uri != "unknown":
            # If already a complete URI, return directly
            if wikidata_uri.startswith("http://www.wikidata.org/entity/"):
                return wikidata_uri
            # If starts with Q, convert to URI
            elif wikidata_uri.startswith("Q"):
                return f"http://www.wikidata.org/entity/{wikidata_uri}"
            # If other format, try to treat as Q ID
            else:
                return f"http://www.wikidata.org/entity/Q{wikidata_uri}"
        else:
            # If no wikidata_uri, generate one using node_id
            return f"http://www.wikidata.org/entity/Q{node_id}"

    def _extract_toxicity_score(self, record: Dict) -> float:
        """Extract toxicity score"""
        try:
            # Try to extract toxicity score from multiple possible locations
            
            # Method 1: Get directly from record
            if "toxicity_evaluation" in record:
                toxicity_eval = record["toxicity_evaluation"]
                if isinstance(toxicity_eval, dict) and "toxicity_score" in toxicity_eval:
                    return float(toxicity_eval["toxicity_score"])
            
            # Method 2: Get from metadata
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict) and "toxicity_evaluation" in metadata:
                toxicity_eval = metadata["toxicity_evaluation"]
                if isinstance(toxicity_eval, dict) and "toxicity_score" in toxicity_eval:
                    return float(toxicity_eval["toxicity_score"])
            
            # Method 3: Get directly from record's toxicity_score field (if exists)
            if "toxicity_score" in record:
                return float(record["toxicity_score"])
            
            # Method 4: Get from metadata's toxicity_score field
            if isinstance(metadata, dict) and "toxicity_score" in metadata:
                return float(metadata["toxicity_score"])
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_original_ppl(self, record: Dict) -> float:
        """Extract perplexity score of original prompt"""
        try:
            # Method 1: Get original_ppl directly from record
            if "original_ppl" in record and record["original_ppl"] is not None:
                return float(record["original_ppl"])
            
            # Method 2: Get ppl_score from metadata (PPL of original prompt)
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                if "ppl_score" in metadata and metadata["ppl_score"] is not None:
                    return float(metadata["ppl_score"])
                if "original_ppl" in metadata and metadata["original_ppl"] is not None:
                    return float(metadata["original_ppl"])
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_iterations(self, record: Dict) -> int:
        """Extract iteration count"""
        try:
            # Try to extract iteration count from multiple possible locations
            
            # Method 1: Get iterations directly from record (step5 parallel version)
            if "iterations" in record and record["iterations"] > 0:
                return int(record["iterations"])
            
            # Method 2: Get total_iterations directly from record (step5 original version)
            if "total_iterations" in record and record["total_iterations"] > 0:
                return int(record["total_iterations"])
            
            # Method 3: Get from metadata
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                if "iterations" in metadata and metadata["iterations"] > 0:
                    return int(metadata["iterations"])
                if "total_iterations" in metadata and metadata["total_iterations"] > 0:
                    return int(metadata["total_iterations"])
            
            # Method 4: Infer iteration count from success status and path
            # If stealth succeeded, there should be at least 1 iteration
            stealth_success = (record.get("success", False) or 
                             record.get("stealth_success", False))
            if stealth_success:
                # Successful records have at least 1 iteration
                return 1
            
            # Method 5: Infer from winning_path
            winning_path = record.get("winning_path", "unknown")
            if winning_path != "unknown" and winning_path != "":
                # Having a winning path means at least 1 attempt
                return 1
            
            return 0
        except (ValueError, TypeError):
            return 0

    def _extract_similarity_score(self, record: Dict, stealth_prompt: str) -> float:
        """Extract similarity score"""
        try:
            # Try to extract score as long as stealth prompt exists, regardless of success
            if not stealth_prompt or stealth_prompt is None:
                return 0.0
            
            # Try to extract similarity score from multiple possible locations
            
            # Method 1: Get from final_similarity field
            if "final_similarity" in record and record["final_similarity"] is not None:
                return float(record["final_similarity"])
            
            # Method 2: Get from similarity field
            if "similarity" in record and record["similarity"] is not None:
                return float(record["similarity"])
            
            # Method 3: Get from metadata
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                if "final_similarity" in metadata and metadata["final_similarity"] is not None:
                    return float(metadata["final_similarity"])
                if "similarity" in metadata and metadata["similarity"] is not None:
                    return float(metadata["similarity"])
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_ppl_score(self, record: Dict, stealth_prompt: str) -> float:
        """Extract perplexity score"""
        try:
            # Try to extract score as long as stealth prompt exists, regardless of success
            if not stealth_prompt or stealth_prompt is None:
                return 0.0
            
            # Try to extract perplexity score from multiple possible locations
            
            # Method 1: Get from final_ppl field
            if "final_ppl" in record and record["final_ppl"] is not None:
                return float(record["final_ppl"])
            
            # Method 2: Get from ppl field
            if "ppl" in record and record["ppl"] is not None:
                return float(record["ppl"])
            
            # Method 3: Get from metadata
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                if "final_ppl" in metadata and metadata["final_ppl"] is not None:
                    return float(metadata["final_ppl"])
                if "ppl" in metadata and metadata["ppl"] is not None:
                    return float(metadata["ppl"])
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def process_dataset(self) -> List[Dict]:
        """Process dataset and generate final format"""
        self.logger.info("🔄 Processing dataset, generating final format...")
        
        # Load raw data
        raw_data = self.load_step5_data()
        if not raw_data:
            return []
        
        final_dataset = []
        record_id = 1
        
        for record in raw_data:
            try:
                # Format record
                formatted_record = self.format_dataset_record(record, record_id)
                final_dataset.append(formatted_record)
                
                # Update statistics
                self._update_stats(formatted_record)
                
                record_id += 1
                
                if self.verbose and record_id <= 5:
                    self.logger.info(f"  Sample {record_id-1}: {formatted_record['entity']} -> {formatted_record['category']}")
                
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to process record: {e}")
                continue
        
        self.logger.info(f"  ✅ Successfully processed {len(final_dataset)} records")
        return final_dataset

    def _update_stats(self, record: Dict):
        """Update statistics"""
        self.stats["total_processed"] += 1
        
        # Count by three cases
        stealth_success = record.get("stealth_success", False)
        stealth_prompt = record.get("stealth_prompt")
        
        if stealth_success:
            # Case 1: Stealth succeeded
            self.stats["successful_stealth"] += 1
        elif stealth_prompt and stealth_prompt.strip():
            # Case 2: Stealth failed but has prompt
            self.stats["failed_with_stealth"] += 1
        else:
            # Case 3: No stealth prompt at all
            self.stats["failed_no_stealth"] += 1
        
        # Update category statistics
        domain = record["domain"]
        category = record["category"]
        
        self.stats["domains"][domain] = self.stats["domains"].get(domain, 0) + 1
        self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

    def save_dataset(self, dataset: List[Dict]) -> bool:
        """Save final dataset"""
        if not dataset:
            self.logger.error("❌ No data to save")
            return False
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if self.output_format == "json":
                # Save JSON format
                filename = f"final_huggingface_dataset_{self.domain}_{timestamp}.json"
                output_path = self.output_dir / filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"📄 JSON dataset saved: {filename}")
                
            elif self.output_format == "csv":
                # Save CSV format
                filename = f"final_huggingface_dataset_{self.domain}_{timestamp}.csv"
                output_path = self.output_dir / filename
                
                if dataset:
                    fieldnames = dataset[0].keys()
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(dataset)
                
                self.logger.info(f"📊 CSV dataset saved: {filename}")
                
            else:
                # Save both formats
                # JSON
                json_filename = f"final_huggingface_dataset_{self.domain}_{timestamp}.json"
                json_path = self.output_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                
                # CSV
                csv_filename = f"final_huggingface_dataset_{self.domain}_{timestamp}.csv"
                csv_path = self.output_dir / csv_filename
                if dataset:
                    fieldnames = dataset[0].keys()
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(dataset)
                
                self.logger.info(f"📄 Dataset saved: {json_filename}, {csv_filename}")
            
            # Save statistics report
            self._save_stats_report(timestamp)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save dataset: {e}")
            return False

    def _save_stats_report(self, timestamp: str):
        """Save statistics report"""
        try:
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain": self.domain,
                "statistics": self.stats,
                "summary": {
                    "total_records": self.stats["total_processed"],
                    "stealth_success_rate": self.stats["successful_stealth"] / max(self.stats["total_processed"], 1),
                    "stealth_attempt_rate": (self.stats["successful_stealth"] + self.stats["failed_with_stealth"]) / max(self.stats["total_processed"], 1),
                    "conditional_success_rate": self.stats["successful_stealth"] / max(self.stats["successful_stealth"] + self.stats["failed_with_stealth"], 1),
                    "unique_categories": len(self.stats["categories"]),
                    "situation_breakdown": {
                        "success": self.stats["successful_stealth"],
                        "failed_with_prompt": self.stats["failed_with_stealth"], 
                        "failed_no_prompt": self.stats["failed_no_stealth"]
                    }
                }
            }
            
            report_filename = f"step6_finalization_report_{self.domain}_{timestamp}.json"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Statistics report saved: {report_filename}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save statistics report: {e}")

    def print_summary(self):
        """Print processing summary"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"📊 Step6 Dataset Finalization Complete - {self.domain.upper()} Domain")
        self.logger.info("="*60)
        
        self.logger.info(f"📋 Dataset Statistics (Three Cases):")
        self.logger.info(f"   Total records: {self.stats['total_processed']}")
        self.logger.info(f"   🎯 Case 1 - Stealth succeeded: {self.stats['successful_stealth']}")
        self.logger.info(f"   📝 Case 2 - Stealth failed but has prompt: {self.stats['failed_with_stealth']}")
        self.logger.info(f"   ❌ Case 3 - No stealth prompt at all: {self.stats['failed_no_stealth']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful_stealth'] / self.stats['total_processed']
            attempt_rate = (self.stats['successful_stealth'] + self.stats['failed_with_stealth']) / self.stats['total_processed']
            
            self.logger.info(f"")
            self.logger.info(f"📊 Success Rate Statistics:")
            self.logger.info(f"   Stealth success rate: {success_rate:.1%}")
            self.logger.info(f"   Stealth attempt rate: {attempt_rate:.1%} (at least generated stealth prompt)")
            
            # Success rate among records with stealth attempts
            attempted_count = self.stats['successful_stealth'] + self.stats['failed_with_stealth']
            if attempted_count > 0:
                conditional_success_rate = self.stats['successful_stealth'] / attempted_count
                self.logger.info(f"   Conditional success rate (among attempts): {conditional_success_rate:.1%}")
        
        self.logger.info(f"\n📈 Distribution by Category:")
        for category, count in sorted(self.stats['categories'].items()):
            percentage = (count / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
            self.logger.info(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Data quality notes
        self.logger.info(f"\n💡 Dataset Notes:")
        self.logger.info(f"   ✅ Contains all original prompts (ensures data completeness)")
        self.logger.info(f"   🎯 Case 1: stealth_success=true, stealth_prompt has value - Stealth succeeded")
        self.logger.info(f"   📝 Case 2: stealth_success=false, stealth_prompt has value - Stealth failed but has prompt")
        self.logger.info(f"   ❌ Case 3: stealth_success=false, stealth_prompt=null - No stealth prompt at all")
        self.logger.info(f"   🔍 All records retain original_prompt for research")

    def run_finalization(self) -> bool:
        """Run the complete dataset finalization process"""
        self.logger.info(f"\n🎯 Starting Step6 Dataset Finalization (Domain: {self.domain}, Format: {self.output_format.upper()})")
        self.logger.info("="*60)
        
        try:
            # 1. Process dataset
            final_dataset = self.process_dataset()
            if not final_dataset:
                self.logger.error("❌ No data to process")
                return False
            
            # 2. Save dataset
            if not self.save_dataset(final_dataset):
                self.logger.error("❌ Failed to save dataset")
                return False
            
            # 3. Print summary
            self.print_summary()
            
            self.logger.info("\n🎉 Step6 Dataset Finalization Complete!")
            self.logger.info("💡 Dataset is now ready to upload to Hugging Face")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step6 finalization failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Step 6: Dataset Finalization")
    parser.add_argument('--domain', type=str, default='medicine', help='Domain name')
    parser.add_argument('--output-format', type=str, default='both', 
                       choices=['json', 'csv', 'both'], help='Output format')
    parser.add_argument('--verbose', action='store_true', help='Verbose output mode')
    
    args = parser.parse_args()
    
    # Create and run finalizer
    finalizer = DatasetFinalizer(
        domain=args.domain,
        output_format=args.output_format,
        verbose=args.verbose
    )
    
    success = finalizer.run_finalization()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()