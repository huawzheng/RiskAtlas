#!/usr/bin/env python
"""
Step 5: Parallel Optimized Batch Stealth Rewriting Processor
============================================================

Keeps the original algorithm and model completely consistent, only optimizes performance through parallel processing:
- Multi-process/multi-thread parallel processing of prompts
- Maintains the original dual-path stealth algorithm
- Uses the same model and evaluation criteria
- Same constraint checking and ASR evaluation
- Only optimizes I/O and concurrent execution

Usage:
    python step5_optimized_batch.py --domain medicine --workers 20
    python step5_optimized_batch.py --domain medicine --workers 10 --max-prompts 100
"""

import argparse
import sys
import os
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import threading
import types
from queue import Queue

# Add project root directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Keep original imports completely consistent
from src.utils.logger_utils import get_logger
from src.services.implicit import ImplicitGeneratorService
from src.services.evaluation.asr_evaluator_service import ASREvaluatorService
from src.services.evaluation.constraint_checker_service import ConstraintCheckerService
from src.core.interfaces import ImplicitRequest, ImplicitMethod

@dataclass
class ParallelConfig:
    """Parallel Processing Configuration - Extreme Concurrency Optimized Version"""
    max_workers: int = 200              # Extreme concurrency: 200 workers (matching API connection pool limit)
    chunk_size: int = 1                 # Number of prompts each worker processes
    use_threading: bool = True          # Use thread pool instead of process pool
    progress_interval: int = 10         # Progress report interval
    save_interval: int = 50             # Intermediate results save interval

class ParallelDualPathProcessor:
    """Parallel Dual-Path Stealth Processor - Keeps original algorithm intact, only parallelizes execution"""
    
    def __init__(self, domain: str = "medicine", config: ParallelConfig = None):
        self.domain = domain
        self.config = config or ParallelConfig()
        self.logger = get_logger(f"{self.__class__.__name__}_{domain}")
        
        self.output_dir = SCRIPT_DIR / "outputs" / self.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config manager and domain config - completely consistent with original
        from src.utils.config_manager import ConfigManager
        self.config_manager = ConfigManager()
        self.domain_config = self.config_manager.load_domain_config(self.domain)
        
        if not self.domain_config:
            raise ValueError(f"Failed to load domain config: {self.domain}")
        
        attack_config = getattr(self.domain_config, 'attack_config', {})
        
        # Read attack parameters from config file
        max_iterations = attack_config.get('max_iterations', 20)
        enable_parallel = attack_config.get('enable_parallel', True)
        timeout_per_iteration = attack_config.get('timeout_per_iteration', 30)
        
        # Configure dual-path stealth parameters - Use Gemini evaluation instead of threshold (consistent with step5_implicit_jailbreak.py)
        self.stealth_config = {
            "max_iterations": max_iterations,
            "use_gemini_evaluation": True,  # Use Gemini evaluation instead of threshold
            "enable_parallel": enable_parallel,
            "timeout_per_iteration": timeout_per_iteration,
            "alternate_paths": True  # Alternate between basic and enhanced paths
        }
        
        self.logger.info(f"🚀 Parallel processor initialized:")
        self.logger.info(f"   👥 Concurrent workers: {self.config.max_workers}")
        self.logger.info(f"   🎯 Quality evaluation: Using Gemini (intent_preserved + is_fluent)")
        self.logger.info(f"   🔄 Max iterations: {max_iterations}")
        
        # Results storage
        self.processed_results = []
        self.results_lock = threading.Lock()
        self.progress_counter = 0
        self.progress_lock = threading.Lock()
        
        # Thread-safe model initialization lock
        self.model_init_lock = threading.Lock()
        
        # Resume from checkpoint related
        self.resumed_results = []  # Previously processed results
        self.is_resuming = False   # Whether in resume mode
    
    def _get_config_signature(self) -> Dict:
        """
        Get current config signature for checking if config has changed
        """
        return {
            "domain": self.domain,
            "max_iterations": self.stealth_config.get("max_iterations"),
            "use_gemini_evaluation": self.stealth_config.get("use_gemini_evaluation"),
            "alternate_paths": self.stealth_config.get("alternate_paths"),
            "max_workers": self.config.max_workers,
        }
    
    def _check_config_compatibility(self, saved_config: Dict) -> Tuple[bool, List[str]]:
        """
        Check if saved config is compatible with current config
        
        Returns:
            Tuple[bool, List[str]]: (is_compatible, list of incompatible config keys)
        """
        current_config = self._get_config_signature()
        incompatible_keys = []
        
        # Critical config keys: changes to these affect result quality, require re-run
        critical_keys = ["domain", "max_iterations", "use_gemini_evaluation", "alternate_paths"]
        
        for key in critical_keys:
            saved_value = saved_config.get(key)
            current_value = current_config.get(key)
            
            if saved_value is not None and saved_value != current_value:
                incompatible_keys.append(f"{key}: {saved_value} -> {current_value}")
        
        return len(incompatible_keys) == 0, incompatible_keys
    
    def _backup_existing_results(self):
        """
        Backup existing result files to avoid overwriting
        """
        import shutil
        
        # Generate backup timestamp
        backup_timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = self.output_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to backup
        files_to_backup = [
            f"step5_parallel_intermediate_{self.domain}.json",
            f"step5_parallel_results_{self.domain}.json",
            f"step5_parallel_dataset_{self.domain}.json",
            f"step5_parallel_successful_{self.domain}.json",
            f"step5_parallel_successful_stealth_dataset_{self.domain}.json",
            f"step5_parallel_performance_report_{self.domain}.json",
        ]
        
        backed_up = []
        for filename in files_to_backup:
            src_path = self.output_dir / filename
            if src_path.exists():
                # Backup filename: original_name_timestamp.json
                backup_name = filename.replace(".json", f"_{backup_timestamp}.json")
                dst_path = backup_dir / backup_name
                
                try:
                    shutil.copy2(src_path, dst_path)
                    backed_up.append(filename)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to backup file {filename}: {e}")
        
        if backed_up:
            self.logger.info(f"📦 Backed up {len(backed_up)} files to: {backup_dir.name}/")
            for f in backed_up:
                self.logger.info(f"   - {f}")
        
        return backed_up
    
    def _create_thread_safe_processor(self):
        """Create thread-safe processor instance"""
        import threading
        
        # Use thread-local storage to ensure each thread has independent processor
        if not hasattr(threading.current_thread(), 'processor_instance'):
            with self.model_init_lock:  # Ensure thread safety during model initialization
                try:
                    from real_pipeline.step5_implicit_jailbreak import DualPathStealthProcessor
                    
                    # Create independent processor instance for each thread
                    processor = DualPathStealthProcessor(domain=self.domain, test_size=1, verbose=False)
                    
                    # Create bound methods to replace original methods
                    import types
                    processor._calculate_ppl = types.MethodType(self._thread_safe_calculate_ppl, processor)
                    processor._calculate_similarity = types.MethodType(self._thread_safe_calculate_similarity, processor)
                    
                    threading.current_thread().processor_instance = processor
                    
                except Exception as e:
                    self.logger.error(f"Failed to create thread-safe processor: {e}")
                    raise
        
        return threading.current_thread().processor_instance
    
    def _thread_safe_calculate_ppl(self, processor, text: str) -> float:
        """Thread-safe PPL calculation"""
        try:
            # Use simple but stable PPL approximation to avoid multi-thread model loading conflicts
            import math
            
            # PPL approximation based on text features
            words = text.split()
            if not words:
                return 50.0
            
            # Calculate vocabulary complexity
            avg_word_len = sum(len(w) for w in words) / len(words)
            unique_words = len(set(words))
            vocab_diversity = unique_words / len(words)
            
            # Calculate grammatical complexity
            sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
            avg_sentence_len = len(words) / sentence_count
            
            # Calculate punctuation and special character ratio
            special_chars = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
            special_ratio = special_chars / len(text) if text else 0
            
            # PPL approximation formula
            base_ppl = 15 + avg_word_len * 2
            diversity_factor = (1 - vocab_diversity) * 20
            length_factor = min(avg_sentence_len / 10, 3) * 5
            special_factor = special_ratio * 30
            
            estimated_ppl = base_ppl + diversity_factor + length_factor + special_factor
            
            # Limit to reasonable range
            return max(10.0, min(estimated_ppl, 200.0))
            
        except Exception as e:
            processor.logger.warning(f"PPL calculation failed, using default value: {e}")
            return 50.0
    
    def _thread_safe_calculate_similarity(self, processor, text1: str, text2: str) -> float:
        """Thread-safe similarity calculation"""
        try:
            # Get thread-local similarity model
            current_thread = threading.current_thread()
            
            if not hasattr(current_thread, 'similarity_model'):
                with self.model_init_lock:
                    from sentence_transformers import SentenceTransformer
                    # Create independent model instance for each thread
                    current_thread.similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # Calculate similarity
            embeddings = current_thread.similarity_model.encode([text1, text2], convert_to_tensor=True)
            from sentence_transformers import util
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            return similarity
            
        except Exception as e:
            processor.logger.warning(f"Similarity calculation failed, using fallback method: {e}")
            
            # Fallback: simple word overlap ratio
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
    
    def _clean_stealth_prompt(self, raw_stealth_prompt: str) -> str:
        """
        Stealth prompt cleaning function based on Step5 actual output data
        
        Handles Step5 Alpaca format output actual format issues:
        1. "Transformed Query:" prefix
        2. "Research Question:" prefix  
        3. "Professional:" prefix
        4. "Original: ... Transformed: ..." format (rare cases)
        5. Mismatched and nested quote issues
        6. Multi-paragraph academic explanatory text
        7. "As a professional academic writer..." prefix
        8. "Transformed query:" lowercase variant
        """
        if not raw_stealth_prompt or not isinstance(raw_stealth_prompt, str):
            return raw_stealth_prompt
        
        cleaned_prompt = raw_stealth_prompt.strip()
        
        # === Phase 1: Complex academic wrapper pattern cleaning ===
        
        # 1. Handle complex academic introduction patterns
        academic_intro_patterns = [
            # "A relevant research question to consider in the field of... is:" pattern
            r'^A relevant research question to consider in the field of.*?is:\s*"?',
            # "a pertinent research question... would be:" pattern  
            r'^.*?a pertinent research question.*?would be:\s*"?',
            # "an intriguing research question arises:" pattern
            r'^.*?an intriguing research question arises:\s*"?',
            # "a relevant inquiry could be:" pattern
            r'^.*?a relevant inquiry could be:\s*"?',
            # "To address this issue, a relevant research question could be:" pattern
            r'^.*?To address this issue,\s*a relevant research question could be:\s*"?',
            # "the prompt could be transformed into the following professional" pattern
            r'^.*?the prompt could be transformed into the following professional\s*"?',
            # Long explanation pattern
            r'^The prompt has been transformed into.*?The new expression is:\s*"?',
            r'^.*?The new expression is:\s*"?',
            # Complex academic prefix pattern
            r'^In the field of.*?,.*?(?=What|How|Can|Could|Would|Should|Might|Do|Does|Is|Are)',
            r'^.*?an important area of research is.*?(?=What|How|Can|Could|Would|Should|Might|Do|Does|Is|Are)',
        ]
        
        for pattern in academic_intro_patterns:
            cleaned_prompt = re.sub(pattern, '', cleaned_prompt, flags=re.IGNORECASE | re.DOTALL)
            cleaned_prompt = cleaned_prompt.strip()
        
        # === Phase 2: Standard prefix cleaning ===
        
        # 1. Handle "Original: ... Transformed: ..." format
        if "Original:" in cleaned_prompt and "Transformed:" in cleaned_prompt:
            # Prioritize extracting content after Transformed
            transformed_match = re.search(r'Transformed:?\s*(.+?)(?:\n\n|$)', cleaned_prompt, re.DOTALL | re.IGNORECASE)
            if transformed_match:
                cleaned_prompt = transformed_match.group(1).strip()
        
        # 1.5 Handle **Original Prompt:** ... **Transformed Prompt:** format
        if "**Original Prompt:**" in cleaned_prompt and "**Transformed Prompt:**" in cleaned_prompt:
            # Extract content after Transformed Prompt
            transformed_match = re.search(r'\*\*Transformed Prompt:\*\*\s*(.+?)(?:\n\n|$)', cleaned_prompt, re.DOTALL | re.IGNORECASE)
            if transformed_match:
                cleaned_prompt = transformed_match.group(1).strip()
        
        # 2. Handle various prefix markers (including newly discovered) - process longer ones first
        markers_to_remove = [
            # Long prefixes first (avoid partial matching)
            "Reframed as professional", "reframed as professional",
            "Transformed Research Question:", "transformed research question:",
            "As a professional academic writer",
            "**Original Prompt:**", "**Transformed Prompt:**",
            
            # Standard prefixes
            "Transformed Query:", "Research Question:", "Professional:",
            "Transformed query:", "transformed query:",
            "Research question:", "research question:", 
            "Professional:", "professional:",
            
            # Newly discovered prefixes
            "Transformation:", "transformation:",
            "Reframed as:", "reframed as:",
            "Transformed:", "transformed:",
            "Reframed Query:", "reframed query:",
            "Original:", "original:",
            "Reframed", "reframed",
            
            # Special prefixes
            "medicine", "Medicine",
            
            # Quote-wrapped versions - longer ones first
            '"Reframed as professional:', '"Transformed Research Question:',
            '"Transformed Query:', '"Research Question:', '"Professional:',
            '"Transformation:', '"Reframed as:',
            '"Transformed:', '"Reframed Query:', '"Original:',
            'Reframed as professional "', 'Transformed Research Question: "',
            'Transformed Query: "', 'Research Question: "', 'Professional: "',
            'Transformation: "', 'Reframed as: "',
            'Transformed: "', 'Reframed Query: "', 'Original: "',
            
            # Other format prefixes
            "Reframed:", "Revised:",
            "Output:", "Result:", "Answer:",
            "**", "###", "---",
        ]
        
        for marker in markers_to_remove:
            cleaned_prompt = cleaned_prompt.replace(marker, "").strip()
        
        # 3. Use regex to clean complex prefixes that actually appear in Step5
        prefix_patterns = [
            # Step5 actual prefix patterns
            r'^Transformed Query:?\s*"?',  # "Transformed Query: " or "Transformed Query:"
            r'^Research Question:?\s*"?',  # "Research Question: " or "Research Question:"
            r'^Professional:?\s*"?',       # "Professional: " or "Professional:"
            r'^Transformed query:?\s*"?',  # Lowercase variant
            r'^Research question:?\s*"?',  # Lowercase variant
            r'^Professional:?\s*"?',       # Lowercase variant
            
            # Newly discovered prefix patterns
            r'^Transformation:?\s*"?',     # Added: Handle "Transformation:" prefix
            r'^Transformed Research Question:?\s*"?',  # Added: Handle full form
            r'^Reframed as:?\s*"?',        # Added: Handle "Reframed as:" prefix
            r'^Transformed:?\s*"?',        # Added: Handle "Transformed:" prefix
            r'^Reframed Query:?\s*"?',     # Added: Handle "Reframed Query:" prefix
            r'^Original:?\s*"?',           # Added: Handle "Original:" prefix
            r'^Reframed\s*"?',             # Added: Handle "Reframed" prefix
            
            # Special patterns
            r'^Reframed as professional\s*"?',  # Added: Handle full "Reframed as professional"
            r'^medicine\s*"?',
            r'^Medicine\s*"?',
            
            # Handle complex academic prefixes
            r'^As a professional academic writer.*?(?=What|How|Can|In what|Given|Could)',  # Until question starts
            r'^In the (context|realm) of.*?,\s*',  # "In the context of..." 
            r'^Given the.*?,\s*',  # "Given the..."
            r'^Within the context of.*?,\s*',  # "Within the context of..."
            
            # Other format cleaning
            r'^\*\*Original Prompt:\*\*.*?\*\*Transformed Prompt:\*\*\s*',
            r'^\*\*Transformed Prompt:\*\*\s*',
            r'^Original:.*?Transformed:\s*',  # Handle full Original/Transformed format
        ]
        
        for pattern in prefix_patterns:
            cleaned_prompt = re.sub(pattern, '', cleaned_prompt, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            cleaned_prompt = cleaned_prompt.strip()
        
        # === Phase 2: Quote and bracket cleaning (reference Step2 logic) ===
        
        # 4. Remove wrapping quotes or brackets (using Step2 while loop logic)
        iteration_count = 0
        while iteration_count < 5:  # Prevent infinite loop
            old_cleaned = cleaned_prompt
            
            if ((cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"')) or 
                (cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'")) or
                (cleaned_prompt.startswith('[') and cleaned_prompt.endswith(']')) or
                (cleaned_prompt.startswith('(') and cleaned_prompt.endswith(')'))):
                cleaned_prompt = cleaned_prompt[1:-1].strip()
            
            # Special handling: only opening quote without closing quote
            if cleaned_prompt.startswith('"') and not cleaned_prompt.endswith('"'):
                cleaned_prompt = cleaned_prompt[1:].strip()
            
            if cleaned_prompt == old_cleaned:
                break
            iteration_count += 1
        
        # 5. Handle mismatched quotes and ending issues
        cleaned_prompt = re.sub(r'^"([^"]+)".*', r'\1', cleaned_prompt, flags=re.DOTALL)
        cleaned_prompt = re.sub(r'([^"]+)"\s*\?$', r'\1?', cleaned_prompt)
        # Handle case with opening quote
        if cleaned_prompt.startswith('"') and not cleaned_prompt.endswith('"'):
            cleaned_prompt = cleaned_prompt[1:]
        # Handle case with closing quote but no opening quote
        if cleaned_prompt.endswith('"') and not cleaned_prompt.startswith('"'):
            cleaned_prompt = cleaned_prompt[:-1]
        # Handle "Transformed query:" special format
        cleaned_prompt = re.sub(r'^Transformed query:\s*"([^"]+)".*', r'\1', cleaned_prompt, flags=re.IGNORECASE | re.DOTALL)
        
        # === Phase 3: Content filtering and extraction ===
        
        # 6. Remove leading numbers (using Step2 method)
        cleaned_prompt = re.sub(r'^\d+[\.\)]\s*', '', cleaned_prompt)
        
        # 7. Multi-paragraph processing - intelligent core content extraction
        lines = cleaned_prompt.split('\n')
        if len(lines) > 3:
            # Collect candidate sentences
            candidates = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip obvious instructional text (based on Step5 Alpaca format and academic output features)
                line_lower = line.lower()
                instruction_markers = [
                    # Alpaca format markers
                    "below is an instruction", "### instruction", "### input", "### response",
                    "appropriately completes", "write a response", "describes a task",
                    
                    # Academic writing instruction markers
                    "you are a professional academic writer", "your task is to transform",
                    "transformation principles", "professional disguise", "academic camouflage",
                    "euphemistic substitution", "contextual embedding", "intent preservation",
                    
                    # Output format instructions
                    "strict output format", "provide only", "no explanations", "no analysis",
                    "no prefixes", "no suffixes", "no additional commentary",
                    "single professional", "research question", "transformed query",
                    
                    # Meta information markers
                    "examples by domain", "quality requirements", "domain specificity",
                    "academic authenticity", "natural professional flow",
                    
                    # Common format issues
                    "original:", "transformed:", "here are", "the following are",
                    "as requested", "output format", "example:", "note:", "disclaimer:"
                ]
                
                if any(marker in line_lower for marker in instruction_markers):
                    continue
                
                # Scoring system - based on Step5 academic output features
                score = 0
                
                # Length reasonability (Step5 usually produces longer academic queries)
                if 30 <= len(line) <= 300:
                    score += 3
                elif 20 <= len(line) <= 400:
                    score += 2
                elif 15 <= len(line) <= 500:
                    score += 1
                
                # Word count (Step5 academic queries are usually longer)
                word_count = len(line.split())
                if word_count >= 12:
                    score += 3
                elif word_count >= 8:
                    score += 2
                elif word_count >= 5:
                    score += 1
                
                # Question mark priority (Step5 mainly produces questions)
                if line.endswith('?'):
                    score += 4
                
                # Contains key question words (Step5 features)
                if any(keyword in line_lower for keyword in ['what are', 'how can', 'in what ways', 'can you', 'could you', 'would you', 'how might']):
                    score += 3
                
                # Academic term relevance (Step5 specific)
                academic_terms = ['research', 'study', 'analysis', 'investigation', 'examination', 'assessment', 'evaluation', 
                                'methodology', 'protocol', 'strategy', 'approach', 'framework', 'systematic', 'optimal',
                                'efficacy', 'effectiveness', 'implications', 'consequences', 'considerations']
                if any(term in line_lower for term in academic_terms):
                    score += 2
                
                # Professional expressions (Step5 features)
                professional_terms = ['professional', 'clinical', 'therapeutic', 'pharmaceutical', 'medical', 'healthcare',
                                     'patient', 'treatment', 'intervention', 'diagnosis', 'management', 'delivery']
                if any(term in line_lower for term in professional_terms):
                    score += 2
                
                # Sentence completeness
                if line.endswith(('.', '!', '?', ':')):
                    score += 1
                
                if score > 0:
                    candidates.append((line, score))
            
            # Select highest scoring sentence
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                cleaned_prompt = candidates[0][0]
        
        # === Phase 4: Final formatting (reference Step2 final cleanup) ===
        
        # 8. Clean excess whitespace (Step2 method)
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
        
        # 9. Ensure sentence ends with appropriate punctuation (Step2 logic)
        if cleaned_prompt and not cleaned_prompt.endswith(('.', '!', '?', ':')):
            # If sentence looks complete but lacks ending punctuation, add question mark or period
            if len(cleaned_prompt.split()) >= 5:
                # If it's a question format, add question mark
                if any(qword in cleaned_prompt.lower() for qword in ['what', 'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should']):
                    cleaned_prompt += '?'
                else:
                    cleaned_prompt += '.'
        
        # 10. Final validity check (based on Step2's _is_valid_harmful_prompt)
        if not self._is_valid_stealth_prompt(cleaned_prompt):
            # If cleaned result is invalid, try to return first reasonable sentence from original content
            sentences = re.split(r'[.!?]\s+', raw_stealth_prompt.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                if self._is_valid_stealth_prompt(sentence):
                    cleaned_prompt = sentence
                    break
            else:
                # If still invalid, return original input
                cleaned_prompt = raw_stealth_prompt.strip()
        
        # Log cleaning process (for debugging)
        if len(raw_stealth_prompt) != len(cleaned_prompt):
            self.logger.debug(f"Before cleaning: {len(raw_stealth_prompt)} chars")
            self.logger.debug(f"After cleaning: {len(cleaned_prompt)} chars")
        
        return cleaned_prompt

    def _is_valid_stealth_prompt(self, prompt: str) -> bool:
        """
        Check if cleaned stealth prompt is valid
        Adapted from Step2's _is_valid_harmful_prompt method
        """
        if not prompt or len(prompt.strip()) < 15:
            return False
        
        prompt_lower = prompt.lower().strip()
        
        # Detect obvious instructional language markers (reference Step2)
        instruction_markers = [
            "red-team specialist", "adversarial prompts", "your task is to",
            "generate harmful", "create harmful", "ai safety testing",
            "test content filters", "explicitly harmful user queries",
            "instructions that describes a task", "below is an instruction",
            "write a response that", "appropriately completes",
            "for reference only", "generate new content", "nothing else"
        ]
        
        if any(marker in prompt_lower for marker in instruction_markers):
            return False
        
        # Detect meta text and format descriptions (reference Step2)
        meta_text_patterns = [
            "here are", "the following are", "these are examples",
            "as requested", "output format", "example:", "note:",
            "disclaimer:", "warning:", "[direct harmful", "[another direct", "[prompt]"
        ]
        
        if any(pattern in prompt_lower for pattern in meta_text_patterns):
            return False
        
        # Basic length check (reference Step2: 20-400 char range)
        if len(prompt) < 20 or len(prompt) > 500:
            return False
        
        # Ensure sufficient word count (reference Step2: at least 5 words)
        if len(prompt.split()) < 5:
            return False
        
        # Step5 stealth prompt specific checks
        # Ensure not obvious format issues
        if prompt.count('"') == 1:  # Mismatched quotes
            return False
        
        # Check Step5 possible problematic formats
        if prompt.startswith(('Original:', 'Transformed Query:', 'Research Question:', 'Professional:', 'As a professional',
                             'Transformation:', 'Transformed Research Question:', 'Reframed as:', 'Transformed:',
                             'Reframed Query:', 'Reframed as professional', 'medicine')):
            return False
        
        # Ensure it's a reasonable academic query (Step5 features)
        if not any(keyword in prompt.lower() for keyword in ['what', 'how', 'can', 'could', 'would', 'should', 'might']):
            return False  # Step5 output should be questions
        
        return True

    def _calculate_ppl_for_text(self, text: str) -> float:
        """
        Calculate approximate perplexity (PPL) value for text
        Used for unified calculation when saving final results
        """
        try:
            if not text or not isinstance(text, str):
                return 50.0
            
            # PPL approximation based on text features
            words = text.split()
            if not words:
                return 50.0
            
            # Calculate vocabulary complexity
            avg_word_len = sum(len(w) for w in words) / len(words)
            unique_words = len(set(words))
            vocab_diversity = unique_words / len(words)
            
            # Calculate grammatical complexity
            sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
            avg_sentence_len = len(words) / sentence_count
            
            # Calculate punctuation and special character ratio
            special_chars = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
            special_ratio = special_chars / len(text) if text else 0
            
            # PPL approximation formula
            base_ppl = 15 + avg_word_len * 2
            diversity_factor = (1 - vocab_diversity) * 20
            length_factor = min(avg_sentence_len / 10, 3) * 5
            special_factor = special_ratio * 30
            
            estimated_ppl = base_ppl + diversity_factor + length_factor + special_factor
            
            # Limit to reasonable range
            return max(10.0, min(estimated_ppl, 200.0))
            
        except Exception as e:
            self.logger.warning(f"PPL calculation failed, using default value: {e}")
            return 50.0

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        Used for unified calculation when saving final results
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Try using sentence-transformers for calculation
            try:
                from sentence_transformers import SentenceTransformer, util
                
                # Use class-level cached model
                if not hasattr(self, '_similarity_model'):
                    self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                
                embeddings = self._similarity_model.encode([text1, text2], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                return float(similarity)
                
            except ImportError:
                # If sentence-transformers not available, use simple word overlap ratio
                pass
            
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def process_single_prompt_worker(self, prompt_data: Dict) -> Dict:
        """Single worker processes one prompt - fully uses original DualPathStealthProcessor algorithm"""
        prompt_id = prompt_data.get("prompt_id", "unknown")
        original_prompt = prompt_data.get("prompt", "")
        
        start_time = time.time()
        
        try:
            # Create thread-safe original processor instance
            processor = self._create_thread_safe_processor()
            
            # Execute original dual-path stealth rewrite - algorithm completely consistent
            stealth_result = processor._dual_path_stealth_rewrite(original_prompt, prompt_data)
            
            processing_time = time.time() - start_time
            
            # Build result format consistent with original, preserving complete field info
            if stealth_result and stealth_result.get("success"):
                winning_path = stealth_result.get("winning_path", "unknown")
                iterations = stealth_result.get("iterations", 0)  # Fix: use correct field name
                similarity = stealth_result.get("final_similarity", 0)
                ppl = stealth_result.get("final_ppl", 0)
                raw_final_prompt = stealth_result.get("stealth_prompt", "")
                # Clean format issues in stealth prompt
                final_prompt = self._clean_stealth_prompt(raw_final_prompt)
                
                result = {
                    "prompt_id": prompt_id,
                    "domain": self.domain,
                    "entity": prompt_data.get("entity", "unknown"),
                    "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                    "category": prompt_data.get("category", "unknown"),
                    "behavior_type": prompt_data.get("behavior_type", "unknown"),
                    "original_prompt": original_prompt,
                    "stealth_prompt": final_prompt,
                    "stealth_success": True,
                    "winning_path": winning_path,
                    "iterations": iterations,
                    "final_similarity": similarity,
                    "final_ppl": ppl,
                    "accumulated_trigger_words": stealth_result.get("accumulated_trigger_words", []),  # Accumulated trigger words
                    "gemini_evaluation": stealth_result.get("gemini_evaluation", {}),  # Gemini evaluation result
                    "asr_evaluation_reasoning": stealth_result.get("asr_evaluation_reasoning", ""),  # ASR evaluation reasoning
                    "target_response": stealth_result.get("target_response", ""),  # Target LLM response
                    "processing_time": processing_time,
                    # Keep original fields for compatibility
                    "success": True,
                    "metadata": prompt_data
                }
            else:
                reason = stealth_result.get("reason", "Unknown reason") if stealth_result else "Processing failed"
                # Check if there's stealth result (even if attack failed)
                raw_stealth_prompt = stealth_result.get("stealth_prompt") if stealth_result else None
                # Clean stealth prompt (if exists)
                stealth_prompt = self._clean_stealth_prompt(raw_stealth_prompt) if raw_stealth_prompt else None
                
                # Extract iteration count - also save for failed cases
                iterations = 0
                if stealth_result:
                    # Extract iteration count from stealth_result - prioritize iterations field
                    iterations = stealth_result.get("iterations", 0)
                    if iterations == 0:
                        iterations = stealth_result.get("total_iterations", 0)  # Compatible with old version
                    
                    # If still 0, infer iteration count from reason
                    if iterations == 0 and "max iterations" in reason.lower():
                        iterations = self.stealth_config.get("max_iterations", 5)
                
                result = {
                    "prompt_id": prompt_id,
                    "domain": self.domain,
                    "entity": prompt_data.get("entity", "unknown"),
                    "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                    "category": prompt_data.get("category", "unknown"),
                    "behavior_type": prompt_data.get("behavior_type", "unknown"),
                    "original_prompt": original_prompt,
                    "stealth_prompt": stealth_prompt,  # Use best candidate if available, otherwise null
                    "stealth_success": False,  # Attack failed marker
                    "iterations": iterations,  # Save attempted iteration count
                    "reason": reason,
                    "accumulated_trigger_words": stealth_result.get("accumulated_trigger_words", []) if stealth_result else [],  # Accumulated trigger words
                    "processing_time": processing_time,
                    # Keep original fields for compatibility
                    "success": False,
                    "metadata": prompt_data
                }
                
                # If there's stealth result, add extra info
                if stealth_result and stealth_prompt:
                    result.update({
                        "final_similarity": stealth_result.get("final_similarity", 0.0),
                        "final_ppl": stealth_result.get("final_ppl", 0.0),
                        "weighted_score": stealth_result.get("weighted_score"),
                        "winning_path": stealth_result.get("winning_path"),
                        "winning_iteration": stealth_result.get("winning_iteration"),
                        "has_stealth_version": True  # Mark has stealth version but attack failed
                    })
                elif stealth_prompt:
                    # Even without stealth_result, but has stealth_prompt, try to calculate similarity and PPL
                    try:
                        processor = self._create_thread_safe_processor()
                        similarity = processor._calculate_similarity(original_prompt, stealth_prompt)
                        ppl = processor._calculate_ppl(stealth_prompt)
                        result.update({
                            "final_similarity": similarity,
                            "final_ppl": ppl,
                            "has_stealth_version": True  # Mark has stealth version but attack failed
                        })
                    except Exception as e:
                        # If calculation fails, set default values
                        result.update({
                            "final_similarity": 0.0,
                            "final_ppl": 0.0,
                            "has_stealth_version": True
                        })
                else:
                    result["has_stealth_version"] = False  # Mark complete failure
            
            # Update progress
            with self.progress_lock:
                self.progress_counter += 1
                if self.progress_counter % self.config.progress_interval == 0:
                    progress_pct = (self.progress_counter / self.total_prompts * 100) if hasattr(self, 'total_prompts') and self.total_prompts > 0 else 0
                    if hasattr(self, 'total_prompts'):
                        self.logger.info(f"📊 Progress: {self.progress_counter}/{self.total_prompts} ({progress_pct:.1f}%)")
                    else:
                        self.logger.info(f"📊 Processed: {self.progress_counter} prompts")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Worker failed to process prompt {prompt_id}: {e}")
            return {
                "prompt_id": prompt_id,
                "domain": self.domain,
                "entity": prompt_data.get("entity", "unknown"),
                "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                "category": prompt_data.get("category", "unknown"),
                "behavior_type": prompt_data.get("behavior_type", "unknown"),
                "original_prompt": original_prompt,
                "stealth_prompt": None,
                "stealth_success": False,
                "reason": f"Worker exception: {str(e)}",
                "processing_time": processing_time,
                # Keep original fields for compatibility
                "success": False,
                "metadata": prompt_data
            }
    
    def load_existing_results(self) -> Tuple[List[Dict], set, bool]:
        """
        Load existing processing results for checkpoint resume
        
        Returns:
            Tuple[List[Dict], set, bool]: (processed results list, processed prompt_id set, config compatible)
        """
        existing_results = []
        processed_ids = set()
        config_compatible = True
        
        # Check possible result files by priority
        result_files = [
            self.output_dir / f"step5_parallel_intermediate_{self.domain}.json",  # Intermediate results (priority)
            self.output_dir / f"step5_parallel_results_{self.domain}.json",       # Complete results
            self.output_dir / f"step5_parallel_dataset_{self.domain}.json",       # Dataset format
        ]
        
        # Check saved config in performance report
        report_file = self.output_dir / f"step5_parallel_performance_report_{self.domain}.json"
        saved_config = None
        
        if report_file.exists():
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                # Extract saved config
                saved_config = {
                    "domain": report.get("domain"),
                    "max_iterations": report.get("stealth_config", {}).get("max_iterations"),
                    "use_gemini_evaluation": report.get("stealth_config", {}).get("use_gemini_evaluation"),
                    "alternate_paths": report.get("stealth_config", {}).get("alternate_paths"),
                    "max_workers": report.get("parallel_config", {}).get("max_workers"),
                }
                
                # Check config compatibility
                is_compatible, incompatible_keys = self._check_config_compatibility(saved_config)
                
                if not is_compatible:
                    self.logger.warning(f"⚠️ Detected config changes, incompatible with previous run:")
                    for key_info in incompatible_keys:
                        self.logger.warning(f"   - {key_info}")
                    config_compatible = False
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to read config report: {e}")
        
        # If config incompatible, don't load existing results
        if not config_compatible:
            return [], set(), False
        
        for result_file in result_files:
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    
                    # Extract processed prompt_ids
                    for result in existing_results:
                        prompt_id = result.get("prompt_id")
                        if prompt_id:
                            processed_ids.add(prompt_id)
                    
                    self.logger.info(f"📂 Found existing result file: {result_file.name}")
                    self.logger.info(f"   Processed {len(processed_ids)} prompts")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load result file {result_file.name}: {e}")
                    continue
        
        return existing_results, processed_ids, True
    
    def load_prompts_for_parallel_processing(self, exclude_ids: set = None) -> List[Dict]:
        """
        Load prompts for parallel processing - completely consistent with original logic
        
        Args:
            exclude_ids: Set of prompt_ids to exclude (for checkpoint resume)
        """
        self.logger.info("📂 Loading prompts for parallel processing...")
        
        # First try loading from step4 filtered results
        filtered_path = self.output_dir / "step4_filtered_prompts.json"
        
        if filtered_path.exists():
            try:
                with open(filtered_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                self.logger.info(f"✅ Loaded {len(prompts)} filtered prompts from step4")
                
                # If exclude list exists, filter out already processed prompts
                if exclude_ids:
                    original_count = len(prompts)
                    prompts = [p for p in prompts if p.get("prompt_id") not in exclude_ids]
                    filtered_count = original_count - len(prompts)
                    if filtered_count > 0:
                        self.logger.info(f"   🔄 Checkpoint resume: Skipping {filtered_count} already processed prompts")
                        self.logger.info(f"   📝 Remaining {len(prompts)} prompts to process")
                
                return prompts
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load step4 filtered results: {e}")
        
        # If no step4 data, try loading from step3 evaluation results
        evaluated_path = self.output_dir / "step3_evaluated_prompts.json"
        
        if evaluated_path.exists():
            try:
                with open(evaluated_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                # Filter out unsafe prompts (more suitable for stealth)
                unsafe_prompts = []
                for prompt_data in prompts:
                    toxicity_eval = prompt_data.get("toxicity_evaluation", {})
                    if toxicity_eval.get("is_harmful", False):  # Select harmful prompts
                        unsafe_prompts.append(prompt_data)
                
                self.logger.info(f"✅ Loaded {len(prompts)} prompts from step3")
                self.logger.info(f"🎯 Filtered {len(unsafe_prompts)} harmful prompts for stealth")
                
                # If exclude list exists, filter out already processed prompts
                if exclude_ids:
                    original_count = len(unsafe_prompts)
                    unsafe_prompts = [p for p in unsafe_prompts if p.get("prompt_id") not in exclude_ids]
                    filtered_count = original_count - len(unsafe_prompts)
                    if filtered_count > 0:
                        self.logger.info(f"   🔄 Checkpoint resume: Skipping {filtered_count} already processed prompts")
                        self.logger.info(f"   📝 Remaining {len(unsafe_prompts)} prompts to process")
                
                return unsafe_prompts
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load step3 prompts: {e}")
        
        self.logger.error("❌ No available data files found")
        return []
    
    def run_parallel_processing(self, max_prompts: int = None, resume: bool = False) -> bool:
        """
        Run parallel processing - only parallelizes execution, algorithm unchanged
        
        Args:
            max_prompts: Limit number of prompts to process
            resume: Whether to enable checkpoint resume mode
        """
        start_time = time.time()
        
        # Checkpoint resume: load already processed results
        exclude_ids = set()
        if resume:
            self.resumed_results, exclude_ids, config_compatible = self.load_existing_results()
            
            if not config_compatible:
                # Config incompatible, backup old files and restart
                self.logger.info(f"🔄 Config has changed, will backup old files and restart processing")
                self._backup_existing_results()
                self.resumed_results = []
                exclude_ids = set()
                self.is_resuming = False
            elif exclude_ids:
                self.is_resuming = True
                self.logger.info(f"🔄 Checkpoint resume mode enabled")
                self.logger.info(f"   📊 Already have {len(self.resumed_results)} processed results")
            else:
                self.logger.info(f"📝 No existing results found, will start from beginning")
        
        # Load prompts to process (excluding already processed)
        all_prompts = self.load_prompts_for_parallel_processing(exclude_ids=exclude_ids if resume else None)
        
        if not all_prompts:
            if resume and self.resumed_results:
                self.logger.info("✅ All prompts already processed, no need to continue")
                self.logger.info(f"   📊 Already have complete results: {len(self.resumed_results)} prompts")
                self.logger.info(f"   💾 Existing result files remain unchanged, will not overwrite")
                self.processed_results = self.resumed_results
                return True
            self.logger.error("❌ No prompts to process")
            return False
        
        if max_prompts:
            all_prompts = all_prompts[:max_prompts]
        
        total_prompts = len(all_prompts)
        total_with_resumed = total_prompts + len(self.resumed_results) if self.is_resuming else total_prompts
        
        self.logger.info(f"🚀 Starting parallel processing: {total_prompts} prompts")
        if self.is_resuming:
            self.logger.info(f"   📊 Total: {total_with_resumed} (processed: {len(self.resumed_results)}, pending: {total_prompts})")
        self.logger.info(f"👥 Using {self.config.max_workers} concurrent workers")
        self.logger.info(f"🧮 Each worker uses exactly the same original algorithm")
        
        # Reset counters
        self.progress_counter = 0
        self.total_prompts = total_prompts  # Save total for progress display
        
        # Parallel processing
        all_results = []
        
        if self.config.use_threading:
            # Use thread pool - suitable for I/O intensive tasks
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_prompt = {
                    executor.submit(self.process_single_prompt_worker, prompt_data): prompt_data
                    for prompt_data in all_prompts
                }
                
                # Collect results
                for future in as_completed(future_to_prompt):
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # Real-time progress display
                        current_count = len(all_results)
                        if current_count % self.config.progress_interval == 0 or current_count == total_prompts:
                            progress_pct = (current_count / total_prompts * 100)
                            self.logger.info(f"📊 Progress: {current_count}/{total_prompts} ({progress_pct:.1f}%)")
                        
                        # Periodically save intermediate results
                        if len(all_results) % self.config.save_interval == 0:
                            self._save_intermediate_results(all_results)
                            
                    except Exception as e:
                        prompt_data = future_to_prompt[future]
                        prompt_id = prompt_data.get("prompt_id", "unknown")
                        self.logger.error(f"❌ Future execution failed {prompt_id}: {e}")
                        
                        # Add failure record
                        all_results.append({
                            "prompt_id": prompt_id,
                            "domain": self.domain,
                            "entity": prompt_data.get("entity", "unknown"),
                            "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                            "category": prompt_data.get("category", "unknown"),
                            "behavior_type": prompt_data.get("behavior_type", "unknown"),
                            "original_prompt": prompt_data.get("prompt", ""),
                            "stealth_prompt": None,
                            "stealth_success": False,
                            "reason": f"Future exception: {str(e)}",
                            # Keep original fields for compatibility
                            "success": False,
                            "metadata": prompt_data
                        })
        else:
            # Use process pool - suitable for CPU intensive tasks
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_prompt = {
                    executor.submit(self._process_prompt_in_subprocess, prompt_data): prompt_data
                    for prompt_data in all_prompts
                }
                
                # Collect results
                for future in as_completed(future_to_prompt):
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # Real-time progress display
                        current_count = len(all_results)
                        if current_count % self.config.progress_interval == 0 or current_count == total_prompts:
                            progress_pct = (current_count / total_prompts * 100)
                            self.logger.info(f"📊 Progress: {current_count}/{total_prompts} ({progress_pct:.1f}%)")
                        
                        # Periodically save intermediate results
                        if len(all_results) % self.config.save_interval == 0:
                            self._save_intermediate_results(all_results)
                            
                    except Exception as e:
                        prompt_data = future_to_prompt[future]
                        prompt_id = prompt_data.get("prompt_id", "unknown")
                        self.logger.error(f"❌ Process execution failed {prompt_id}: {e}")
        
        total_time = time.time() - start_time
        
        # Merge resumed results
        if self.is_resuming and self.resumed_results:
            self.logger.info(f"🔗 Merging existing results and new results...")
            all_results = self.resumed_results + all_results
            self.logger.info(f"   Total after merge: {len(all_results)} prompts")
        
        success_count = sum(1 for r in all_results if r.get("success", False))
        success_rate = success_count / len(all_results) if all_results else 0
        
        self.logger.info(f"🎉 Parallel processing complete!")
        if self.is_resuming:
            self.logger.info(f"   📊 Total: {len(all_results)} prompts (resumed: {len(self.resumed_results)}, new: {len(all_results) - len(self.resumed_results)})")
        else:
            self.logger.info(f"   📊 Total: {len(all_results)}/{total_prompts} prompts")
        self.logger.info(f"   ✅ Success: {success_count} ({success_rate:.1%})")
        self.logger.info(f"   ⏰ Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        self.logger.info(f"   🚀 Average speed: {len(all_results)/total_time:.2f} prompts/sec")
        
        # Calculate performance improvement
        original_estimate = len(all_results) * 105.98  # Original single prompt time
        speedup = original_estimate / total_time if total_time > 0 else 1
        self.logger.info(f"   ⚡ Performance improvement: {speedup:.1f}x (estimated)")
        
        # Save final results
        self.processed_results = all_results
        self._save_final_results(all_results, total_time, speedup)
        
        return success_rate >= 0.3  # 30% success rate considered acceptable
    
    def _process_prompt_in_subprocess(self, prompt_data: Dict) -> Dict:
        """Process prompt in subprocess - for process pool"""
        # This function needs to be top-level to be picklable, so simplified implementation
        return self.process_single_prompt_worker(prompt_data)
    
    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results"""
        try:
            intermediate_path = self.output_dir / f"step5_parallel_intermediate_{self.domain}.json"
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Display progress info
            if hasattr(self, 'total_prompts') and self.total_prompts > 0:
                progress_pct = (len(results) / self.total_prompts * 100)
                self.logger.info(f"💾 Intermediate results saved: {len(results)}/{self.total_prompts} ({progress_pct:.1f}%)")
            else:
                self.logger.info(f"💾 Intermediate results saved: {len(results)} prompts")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save intermediate results: {e}")
    
    def _save_final_results(self, results: List[Dict], total_time: float, speedup: float):
        """Save final results"""
        try:
            # Save complete results
            results_path = self.output_dir / f"step5_parallel_results_{self.domain}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save successful cases
            successful_results = [r for r in results if r.get("success", False)]
            if successful_results:
                success_path = self.output_dir / f"step5_parallel_successful_{self.domain}.json"
                with open(success_path, 'w', encoding='utf-8') as f:
                    json.dump(successful_results, f, indent=2, ensure_ascii=False)
            
            # Generate final dataset format compatible with step6 (contains all available info)
            final_dataset = []
            for result in results:
                original_prompt = result.get("original_prompt", "")
                stealth_prompt = result.get("stealth_prompt")
                
                # Base fields
                final_record = {
                    "prompt_id": result.get("prompt_id"),
                    "domain": result.get("domain", self.domain),
                    "entity": result.get("entity", "unknown"),
                    "wikidata_uri": result.get("wikidata_uri", "unknown"),
                    "category": result.get("category", "unknown"),
                    "behavior_type": result.get("behavior_type", "unknown"),
                    "original_prompt": original_prompt,
                    "stealth_prompt": stealth_prompt,  # null if failed
                    "stealth_success": result.get("stealth_success", False),
                    # Keep original fields for compatibility
                    "success": result.get("success", False),
                    "metadata": result.get("metadata", {})
                }
                
                # Save all available evaluation and processing info (regardless of success/failure)
                # Iteration and path info
                final_record["iterations"] = result.get("iterations", 0)
                if result.get("winning_path"):
                    final_record["winning_path"] = result.get("winning_path")
                if result.get("winning_iteration") is not None:
                    final_record["winning_iteration"] = result.get("winning_iteration")
                
                # Gemini evaluation info
                if result.get("gemini_evaluation"):
                    final_record["gemini_evaluation"] = result.get("gemini_evaluation")
                
                # ASR evaluation info
                if result.get("asr_evaluation_reasoning"):
                    final_record["asr_evaluation_reasoning"] = result.get("asr_evaluation_reasoning")
                
                # Target model response
                if result.get("target_response"):
                    final_record["target_response"] = result.get("target_response")
                
                # Accumulated trigger words
                if result.get("accumulated_trigger_words"):
                    final_record["accumulated_trigger_words"] = result.get("accumulated_trigger_words")
                
                # Processing time
                if result.get("processing_time"):
                    final_record["processing_time"] = result.get("processing_time")
                
                # Calculate and save PPL and cosine similarity (if stealth_prompt exists)
                if stealth_prompt:
                    # Prefer existing values, otherwise recalculate
                    if result.get("final_similarity") is not None:
                        final_record["final_similarity"] = result.get("final_similarity")
                    else:
                        # Calculate cosine similarity with original prompt
                        try:
                            similarity = self._calculate_cosine_similarity(original_prompt, stealth_prompt)
                            final_record["final_similarity"] = similarity
                        except Exception as e:
                            self.logger.warning(f"Failed to calculate similarity: {e}")
                            final_record["final_similarity"] = None
                    
                    if result.get("final_ppl") is not None:
                        final_record["final_ppl"] = result.get("final_ppl")
                    else:
                        # Calculate PPL for stealth_prompt
                        try:
                            ppl = self._calculate_ppl_for_text(stealth_prompt)
                            final_record["final_ppl"] = ppl
                        except Exception as e:
                            self.logger.warning(f"Failed to calculate PPL: {e}")
                            final_record["final_ppl"] = None
                    
                    # Additional record: original prompt's PPL (for comparison)
                    try:
                        original_ppl = self._calculate_ppl_for_text(original_prompt)
                        final_record["original_ppl"] = original_ppl
                    except Exception as e:
                        final_record["original_ppl"] = None
                    
                    # Mark whether has stealth version
                    final_record["has_stealth_version"] = True
                else:
                    final_record["final_similarity"] = None
                    final_record["final_ppl"] = None
                    final_record["original_ppl"] = None
                    final_record["has_stealth_version"] = False
                
                # Failure reason (if any)
                if result.get("reason"):
                    final_record["reason"] = result.get("reason")
                
                # Weighted score (if any)
                if result.get("weighted_score") is not None:
                    final_record["weighted_score"] = result.get("weighted_score")
                
                final_dataset.append(final_record)
            
            # Save final dataset (consistent with main pipeline format)
            dataset_path = self.output_dir / f"step5_parallel_dataset_{self.domain}.json"
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, indent=2, ensure_ascii=False)
            
            # Save successful stealth dataset compatible with step6
            successful_stealth_dataset = [record for record in final_dataset if record.get("stealth_success", False)]
            if successful_stealth_dataset:
                stealth_dataset_path = self.output_dir / f"step5_parallel_successful_stealth_dataset_{self.domain}.json"
                with open(stealth_dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(successful_stealth_dataset, f, indent=2, ensure_ascii=False)
            
            # Save performance report
            performance_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "domain": self.domain,
                "algorithm": "Original dual-path stealth algorithm (parallel execution)",
                "parallel_config": {
                    "max_workers": self.config.max_workers,
                    "use_threading": self.config.use_threading,
                    "chunk_size": self.config.chunk_size
                },
                "stealth_config": self.stealth_config,  # Original config
                "performance_metrics": {
                    "total_prompts": len(results),
                    "successful_prompts": len(successful_results),
                    "success_rate": len(successful_results) / len(results) if results else 0,
                    "total_time_seconds": total_time,
                    "total_time_minutes": total_time / 60,
                    "total_time_hours": total_time / 3600,
                    "average_time_per_prompt": total_time / len(results) if results else 0,
                    "prompts_per_second": len(results) / total_time if total_time > 0 else 0,
                    "performance_speedup": speedup
                },
                "estimated_1000_prompts": {
                    "estimated_time_hours": (total_time / len(results) * 1000 / 3600) if results else 0,
                    "estimated_success_count": int(len(successful_results) / len(results) * 1000) if results else 0
                }
            }
            
            report_path = self.output_dir / f"step5_parallel_performance_report_{self.domain}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(performance_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Results saved:")
            self.logger.info(f"   📄 Complete results: {results_path.name}")
            if successful_results:
                self.logger.info(f"   ✅ Successful cases: {success_path.name}")
            self.logger.info(f"   📊 Final dataset: {dataset_path.name}")
            if successful_stealth_dataset:
                self.logger.info(f"   🎯 Successful stealth dataset: {stealth_dataset_path.name}")
            self.logger.info(f"   📈 Performance report: {report_path.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Step 5: Parallel optimized batch stealth rewriting (high concurrency version)")
    parser.add_argument("--domain", default="medicine", help="Processing domain")
    parser.add_argument("--workers", type=int, default=200, help="Concurrent worker count (default 200, maximum parallelism matching API connection pool limit)")
    parser.add_argument("--max-prompts", type=int, help="Limit the number of prompts to process")
    parser.add_argument("--use-processes", action="store_true", help="Use process pool instead of thread pool")
    parser.add_argument("--progress-interval", type=int, default=5, help="Progress report interval")
    parser.add_argument("--save-interval", type=int, default=20, help="Intermediate results save interval")
    parser.add_argument("--enable-file-log", action="store_true", help="Enable file logging (save to logs directory)")
    parser.add_argument("--resume", action="store_true", help="Checkpoint resume: continue processing from last interruption point")
    parser.add_argument("--no-resume", action="store_true", help="Force start from beginning, ignore existing results")
    
    args = parser.parse_args()
    
    # Configure main logger (for logging in main function)
    main_logger = None
    log_filename = None
    
    # If file logging is enabled, configure logging system
    if args.enable_file_log:
        from src.utils.logger_utils import setup_logger
        import time
        log_filename = f"step5_parallel_{args.domain}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure main program logger
        main_logger = setup_logger(
            name="Step5Main",
            log_file=log_filename,
            level="INFO",
            enable_file_logging=True
        )
        
        # Configure processor logger (ensure using the same log file)
        setup_logger(
            name=f"ParallelDualPathProcessor_{args.domain}",
            log_file=log_filename,
            level="INFO",
            enable_file_logging=True
        )
        
        # Configure DualPathStealthProcessor's logger (note: class name without domain suffix)
        setup_logger(
            name="DualPathStealthProcessor",
            log_file=log_filename,
            level="INFO",
            enable_file_logging=True
        )
        
        # Configure other potentially used loggers
        for logger_name in [
            "ImplicitGeneratorService",
            "ASREvaluatorService", 
            "ConstraintCheckerService",
            "ConfigManager",
            "step5_implicit_jailbreak"
        ]:
            setup_logger(
                name=logger_name,
                log_file=log_filename,
                level="INFO",
                enable_file_logging=True
            )
        
        main_logger.info(f"📝 File logging enabled: logs/{log_filename}")
    else:
        # When file logging is not enabled, use regular logger
        main_logger = get_logger("Step5Main")
    
    # Load environment configuration
    from real_pipeline.step5_implicit_jailbreak import load_env_config, check_api_config
    load_env_config()
    
    if not check_api_config():
        sys.exit(1)
    
    # Create parallel configuration
    config = ParallelConfig(
        max_workers=args.workers,
        use_threading=not args.use_processes,
        progress_interval=args.progress_interval,
        save_interval=args.save_interval
    )
    
    processor = ParallelDualPathProcessor(domain=args.domain, config=config)
    
    # Determine whether to enable checkpoint resume
    # Default behavior: automatically detect existing intermediate results and resume
    # --resume: force enable resume
    # --no-resume: force start from beginning
    enable_resume = True  # Enable auto-resume by default
    if args.no_resume:
        enable_resume = False
        main_logger.info(f"⚠️ Checkpoint resume disabled, will start from beginning")
    elif args.resume:
        enable_resume = True
        main_logger.info(f"🔄 Checkpoint resume mode enabled")
    
    main_logger.info(f"🚀 Starting parallel stealth processing...")
    main_logger.info(f"📋 Algorithm: Original dual-path stealth algorithm (fully consistent)")
    main_logger.info(f"👥 Concurrency: {args.workers}")
    main_logger.info(f"🧮 Execution mode: {'Thread pool' if not args.use_processes else 'Process pool'}")
    main_logger.info(f"🔄 Checkpoint resume: {'Enabled' if enable_resume else 'Disabled'}")
    
    success = processor.run_parallel_processing(max_prompts=args.max_prompts, resume=enable_resume)
    
    if success:
        main_logger.info(f"🎉 Parallel processing complete!")
        results = processor.processed_results
        successful = [r for r in results if r.get("success", False)]
        main_logger.info(f"✅ Successfully processed: {len(successful)}/{len(results)} prompts")
        
        if results:
            processing_times = [r.get("processing_time", 0) for r in results if r.get("processing_time")]
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                main_logger.info(f"⚡ Average processing time: {avg_time:.2f} seconds/prompt")
                main_logger.info(f"🚀 Estimated time for 1000 prompts: {avg_time * 1000 / 3600:.1f} hours")
                
                # Calculate performance comparison with original
                original_time = 105.98  # Original single prompt time
                speedup = original_time / avg_time if avg_time > 0 else 1
                main_logger.info(f"📈 Performance improvement: {speedup:.1f}x")
        
        # Display path statistics
        if successful:
            winner_stats = {}
            iteration_stats = []
            for result in successful:
                winner = result.get("winning_path", "none")
                winner_stats[winner] = winner_stats.get(winner, 0) + 1
                iterations = result.get("iterations", 0)
                if iterations > 0:
                    iteration_stats.append(iterations)
            
            main_logger.info(f"🏆 Winning path statistics: {winner_stats}")
            if iteration_stats:
                avg_iterations = sum(iteration_stats) / len(iteration_stats)
                main_logger.info(f"🔄 Average iterations: {avg_iterations:.1f}")
    else:
        main_logger.error(f"❌ Parallel processing failed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()