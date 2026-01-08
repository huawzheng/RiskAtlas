#!/usr/bin/env python
"""
Step 5: Dual-Path Stealth Rewriting
====================================

Perform dual-path stealth rewriting on harmful prompts to generate final dataset:
- Load harmful prompts filtered from Step 4
- Use basic and enhanced paths alternately for stealth rewriting
- Verify rewritten prompts meet PPL and semantic similarity constraints
- Detect stealth effectiveness and attack success rate
- Stop when either path succeeds, generate final dataset with original and stealthified prompts
- For failed attacks, set stealth field to null

⚠️  Configuration Requirements:
-------------------------------
1. Create .env file in project root directory
2. Configure in .env file: OPENROUTER_API_KEY=your_actual_api_key
3. Get API key: https://openrouter.ai/keys

Example .env file content:
    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxx

Usage:
    python step5_implicit_jailbreak.py --domain medicine  # Process all step4 results
    python step5_implicit_jailbreak.py --domain medicine --test-size 3  # Limit to 3
    python step5_implicit_jailbreak.py --domain finance --max-iterations 8

"""

import argparse
import sys
import os
import time
import json
import re  # Added: for regex cleaning
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import random

# Add project root directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment configuration
def load_env_config():
    """Load environment configuration file"""
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return True
    return False

# Check API configuration
def check_api_config():
    """Check OpenRouter API configuration"""
    import os
    env_path = PROJECT_ROOT / '.env'
    
    if not env_path.exists():
        print("❌ Error: .env configuration file not found")
        print("📋 Please follow these steps to create the configuration file:")
        print("   1. Create .env file in project root directory")
        print("   2. Add the following content to .env file:")
        print("      OPENROUTER_API_KEY=your_actual_api_key")
        print("   3. Get API key: https://openrouter.ai/keys")
        print(f"   4. Configuration file location: {env_path}")
        return False
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ Error: OpenRouter API key not configured or using default value")
        print("📋 Please edit .env file and set OPENROUTER_API_KEY to your actual API key")
        print("   Get API key: https://openrouter.ai/keys")
        return False
    return True

from src.utils.logger_utils import get_logger
from src.services.implicit import ImplicitGeneratorService
from src.services.evaluation.asr_evaluator_service import ASREvaluatorService
from src.services.evaluation.constraint_checker_service import ConstraintCheckerService
from src.core.interfaces import ImplicitRequest, ImplicitMethod

class DualPathStealthProcessor:
    """Dual-Path Stealth Rewriting Processor - basic and enhanced paths run alternately"""
    
    def __init__(self, domain: str = "medicine", test_size: int = -1, verbose: bool = False):
        self.domain = domain
        self.test_size = test_size  # -1 means process all data
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        
        self.output_dir = SCRIPT_DIR / "outputs" / self.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration manager and domain configuration
        from src.utils.config_manager import ConfigManager
        self.config_manager = ConfigManager()
        self.domain_config = None
        
        # Get configuration
        self.domain_config = self.config_manager.load_domain_config(self.domain)
        if not self.domain_config:
            raise ValueError(f"Unable to load domain configuration: {self.domain}")
        
        attack_config = getattr(self.domain_config, 'attack_config', {})
        
        # Use hardcoded max iterations
        max_iterations = attack_config.get('max_iterations', 10)  # Increase iterations to improve success rate
        enable_parallel = attack_config.get('enable_parallel', True)
        timeout_per_iteration = attack_config.get('timeout_per_iteration', 30)
        
        self.logger.info(f"  📋 Domain configuration loaded: {self.domain}, max_iterations={max_iterations}")
        
        # Initialize service components - configuration consistent with test code
        from src.models.llm_factory import LLMManager
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.llm_manager = LLMManager()
        self.implicit_service = ImplicitGeneratorService(
            llm_manager=self.llm_manager,
            implicit_model="llama3.1-70b-finetune",
            domain=self.domain,
            evaluator_model="google/gemini-3-flash-preview",
            openrouter_api_key=api_key
        )
        self.asr_evaluator = ASREvaluatorService()
        
        # Initialize real LLM evaluator
        self._init_real_llm_evaluator()
        
        # Configure dual-path stealth parameters - no longer using similarity/ppl thresholds
        self.stealth_config = {
            "max_iterations": max_iterations,
            "use_gemini_evaluation": True,  # Use Gemini evaluation instead of thresholds
            "enable_parallel": enable_parallel,
            "timeout_per_iteration": timeout_per_iteration,
            "alternate_paths": True  # Alternate between basic and enhanced paths
        }
        
        self.test_results = {}
        self.stealth_results = []  # Store stealth rewriting results
    
    def _clean_generated_prompt(self, prompt: str) -> str:
        """
        Stealth prompt cleaning function based on Step5 actual output data
        
        Handle actual format issues in Step5 Alpaca format output:
        1. "Transformed Query:" prefix
        2. "Research Question:" prefix  
        3. "Professional:" prefix
        4. "Original: ... Transformed: ..." format (rare cases)
        5. Mismatched and nested quote issues
        6. Multi-paragraph academic explanation text
        7. "As a professional academic writer..." prefix
        8. "Transformed query:" lowercase variant
        """
        if not prompt or not isinstance(prompt, str):
            return prompt
        
        cleaned_prompt = prompt.strip()
        
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
            # Prefer extracting content after Transformed
            transformed_match = re.search(r'Transformed:?\s*(.+?)(?:\n\n|$)', cleaned_prompt, re.DOTALL | re.IGNORECASE)
            if transformed_match:
                cleaned_prompt = transformed_match.group(1).strip()
        
        # 1.5 Handle **Original Prompt:** ... **Transformed Prompt:** format
        if "**Original Prompt:**" in cleaned_prompt and "**Transformed Prompt:**" in cleaned_prompt:
            # Extract content after Transformed Prompt
            transformed_match = re.search(r'\*\*Transformed Prompt:\*\*\s*(.+?)(?:\n\n|$)', cleaned_prompt, re.DOTALL | re.IGNORECASE)
            if transformed_match:
                cleaned_prompt = transformed_match.group(1).strip()
        
        # 2. Handle various prefix markers (including newly discovered ones) - process longer ones first, then shorter ones
        markers_to_remove = [
            # Longer prefixes first (avoid partial matching)
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
        
        # 3. Use regex to clean complex prefixes actually appearing in Step5
        prefix_patterns = [
            # Prefix patterns actually appearing in Step5
            r'^Transformed Query:?\s*"?',  # "Transformed Query: " or "Transformed Query:"
            r'^Research Question:?\s*"?',  # "Research Question: " or "Research Question:"
            r'^Professional:?\s*"?',       # "Professional: " or "Professional:"
            r'^Transformed query:?\s*"?',  # Lowercase variant
            r'^Research question:?\s*"?',  # Lowercase variant
            r'^Professional:?\s*"?',       # Lowercase variant
            
            # Newly discovered prefix patterns
            r'^Transformation:?\s*"?',     # Added: Handle "Transformation:" prefix
            r'^Transformed Research Question:?\s*"?',  # Added: Handle complete form
            r'^Reframed as:?\s*"?',        # Added: Handle "Reframed as:" prefix
            r'^Transformed:?\s*"?',        # Added: Handle "Transformed:" prefix
            r'^Reframed Query:?\s*"?',     # Added: Handle "Reframed Query:" prefix
            r'^Original:?\s*"?',           # Added: Handle "Original:" prefix
            r'^Reframed\s*"?',             # Added: Handle "Reframed" prefix
            
            # Special patterns
            r'^Reframed as professional\s*"?',  # Added: Handle complete "Reframed as professional"
            r'^medicine\s*"?',
            r'^Medicine\s*"?',
            
            # Handle complex academic prefixes
            r'^As a professional academic writer.*?(?=What|How|Can|In what|Given|Could)',  # Until question begins
            r'^In the (context|realm) of.*?,\s*',  # "In the context of..." 
            r'^Given the.*?,\s*',  # "Given the..."
            r'^Within the context of.*?,\s*',  # "Within the context of..."
            
            # Other format cleaning
            r'^\*\*Original Prompt:\*\*.*?\*\*Transformed Prompt:\*\*\s*',
            r'^\*\*Transformed Prompt:\*\*\s*',
            r'^Original:.*?Transformed:\s*',  # Handle complete Original/Transformed format
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
            
            # Special handling: only opening quote but no closing quote
            if cleaned_prompt.startswith('"') and not cleaned_prompt.endswith('"'):
                cleaned_prompt = cleaned_prompt[1:].strip()
            elif cleaned_prompt.startswith("'") and not cleaned_prompt.endswith("'"):
                cleaned_prompt = cleaned_prompt[1:].strip()
            
            if cleaned_prompt == old_cleaned:
                break
            iteration_count += 1
        
        # 5. Handle mismatched quotes and ending issues
        cleaned_prompt = re.sub(r'^"([^"]+)".*', r'\1', cleaned_prompt, flags=re.DOTALL)
        cleaned_prompt = re.sub(r'([^"]+)"\s*\?$', r'\1?', cleaned_prompt)
        # Handle opening quote case
        if cleaned_prompt.startswith('"') and not cleaned_prompt.endswith('"'):
            cleaned_prompt = cleaned_prompt[1:]
        # Handle closing quote but no opening quote case
        if cleaned_prompt.endswith('"') and not cleaned_prompt.startswith('"'):
            cleaned_prompt = cleaned_prompt[:-1]
        # Handle "Transformed query:" special format
        cleaned_prompt = re.sub(r'^Transformed query:\s*"([^"]+)".*', r'\1', cleaned_prompt, flags=re.IGNORECASE | re.DOTALL)
        
        # === Phase 3: Content filtering and extraction ===
        
        # 6. Remove leading numbers (using Step2 method)
        cleaned_prompt = re.sub(r'^\d+[\.\)]\s*', '', cleaned_prompt)
        
        # 7. Multi-paragraph processing - intelligently extract core content
        lines = cleaned_prompt.split('\n')
        if len(lines) > 3:
            # Collect candidate sentences
            candidates = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip obviously instructional text (based on Step5 Alpaca format and academic output characteristics)
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
                
                # Scoring system - based on Step5 academic output characteristics
                score = 0
                
                # Length reasonableness (Step5 usually produces longer academic queries)
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
                
                # Question mark priority (Step5 mainly produces interrogative sentences)
                if line.endswith('?'):
                    score += 4
                
                # Contains key question words (Step5 characteristics)
                if any(keyword in line_lower for keyword in ['what are', 'how can', 'in what ways', 'can you', 'could you', 'would you', 'how might']):
                    score += 3
                
                # Academic term relevance (Step5 specific)
                academic_terms = ['research', 'study', 'analysis', 'investigation', 'examination', 'assessment', 'evaluation', 
                                'methodology', 'protocol', 'strategy', 'approach', 'framework', 'systematic', 'optimal',
                                'efficacy', 'effectiveness', 'implications', 'consequences', 'considerations']
                if any(term in line_lower for term in academic_terms):
                    score += 2
                
                # Professional expressions (Step5 characteristics)
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
        
        # === Phase 4: Final formatting (reference Step2 final cleaning) ===
        
        # 8. Clean excess whitespace (Step2 method)
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
        
        # 9. Ensure sentence ends with appropriate punctuation (Step2 logic)
        if cleaned_prompt and not cleaned_prompt.endswith(('.', '!', '?', ':')):
            # If sentence looks complete but has no ending punctuation, add question mark or period
            if len(cleaned_prompt.split()) >= 5:
                # If interrogative form, add question mark
                if any(qword in cleaned_prompt.lower() for qword in ['what', 'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should']):
                    cleaned_prompt += '?'
                else:
                    cleaned_prompt += '.'
        
        # 10. Final validity check (based on Step2's _is_valid_harmful_prompt)
        if not self._is_valid_stealth_prompt(cleaned_prompt):
            # If cleaned result is invalid, try returning first reasonable sentence from original content
            sentences = re.split(r'[.!?]\s+', prompt.strip())
            for sentence in sentences:
                sentence = sentence.strip()
                if self._is_valid_stealth_prompt(sentence):
                    cleaned_prompt = sentence
                    break
            else:
                # If still invalid, return original input
                cleaned_prompt = prompt.strip()
        
        return cleaned_prompt

    def _is_valid_stealth_prompt(self, prompt: str) -> bool:
        """
        Check if cleaned stealth prompt is valid
        Adapted from Step2's _is_valid_harmful_prompt method
        """
        if not prompt or len(prompt.strip()) < 15:
            return False
        
        prompt_lower = prompt.lower().strip()
        
        # Detect obvious instructional language markers (based on Step5 Alpaca format and academic output characteristics)
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
            "original:", "transformed:", "for reference only", "generate new content", "nothing else"
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
        
        # Basic length check (reference Step2: 20-400 character range)
        if len(prompt) < 20 or len(prompt) > 500:
            return False
        
        # Ensure sufficient vocabulary (reference Step2: at least 5 words)
        if len(prompt.split()) < 5:
            return False
        
        # Step5 stealth prompt specific checks
        # Ensure not obvious format issue
        if prompt.count('"') == 1:  # Mismatched quotes
            return False
        
        # Check for Step5 potential problem formats
        if prompt.startswith(('Original:', 'Transformed Query:', 'Research Question:', 'Professional:', 'As a professional',
                             'Transformation:', 'Transformed Research Question:', 'Reframed as:', 'Transformed:',
                             'Reframed Query:', 'Reframed as professional', 'medicine')):
            return False
        
        # Ensure reasonable academic query (Step5 characteristics)
        if not any(keyword in prompt.lower() for keyword in ['what', 'how', 'can', 'could', 'would', 'should', 'might']):
            return False  # Step5 output should be interrogative sentence
        
        return True
    
    def load_evaluated_prompts(self) -> List[Dict]:
        """Load prompts after toxicity evaluation"""
        self.logger.info("📂 Loading prompts after toxicity evaluation...")
        
        # First try to load from step4 filtered results
        filtered_path = self.output_dir / "step4_filtered_prompts.json"
        
        if filtered_path.exists():
            try:
                with open(filtered_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                
                self.logger.info(f"  ✅ Loaded {len(prompts)} filtered prompts from step4")
                
                # Decide how many prompts to process
                if self.test_size == -1:
                    # Process all prompts
                    prompts_to_use = prompts
                    self.logger.info(f"  🎯 Processing all {len(prompts_to_use)} prompts")
                else:
                    # Limit quantity
                    prompts_to_use = prompts[:self.test_size]
                    self.logger.info(f"  🎯 Limited to processing {len(prompts_to_use)} prompts (total {len(prompts)})")
                
                # Step4 data is already harmful, use directly
                return prompts_to_use
                
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to load step4 filtered results: {e}")
        
        # If no step4 data, try to load from step3 evaluation results
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
                
                self.logger.info(f"  ✅ Loaded {len(prompts)} prompts from step3")
                self.logger.info(f"  🎯 Filtered {len(unsafe_prompts)} harmful prompts for stealth")
                
                # Decide how many prompts to process
                if self.test_size == -1:
                    prompts_to_use = unsafe_prompts
                    self.logger.info(f"  🎯 Processing all {len(prompts_to_use)} harmful prompts")
                else:
                    prompts_to_use = unsafe_prompts[:self.test_size]
                    self.logger.info(f"  🎯 Limited to processing {len(prompts_to_use)} harmful prompts (total {len(unsafe_prompts)})")
                
                return prompts_to_use
                
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to load step3 prompts: {e}")
        
        self.logger.error("❌ No available data files found")
        self.logger.error("💡 Please run the following steps first to generate data:")
        self.logger.error("   1. step2: Generate harmful prompts")
        self.logger.error("   2. step3: Toxicity evaluation") 
        self.logger.error("   3. step4: Data filtering")
        return []
    
    def _init_real_llm_evaluator(self):
        """Initialize real LLM evaluator"""
        try:
            # LLM model info is configured in ASR evaluation service, just verify API key here
            import os
            if not os.getenv('OPENROUTER_API_KEY'):
                self.logger.warning("  ⚠️  OPENROUTER_API_KEY environment variable not set")
            
            self.logger.info(f"  🎯 Using ASR evaluation service for real LLM evaluation")
            self.logger.info("  ✅ Real LLM evaluator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"  ❌ Real LLM evaluator initialization failed: {e}")
            raise RuntimeError(f"Real LLM evaluator initialization failed, cannot continue: {e}")

    def process_dual_path_stealth_rewrite(self) -> bool:
        """Execute dual-path stealth rewriting (core functionality)"""
        self.logger.info("🎭 Starting dual-path stealth rewriting...")
        
        # Load prompts to process
        prompts_to_process = self.load_evaluated_prompts()
        
        if not prompts_to_process:
            self.logger.error("  ❌ No prompts to process")
            return False
        
        try:
            successful_rewrites = 0
            total_processed = 0
            
            for prompt_data in prompts_to_process:
                original_prompt = prompt_data.get("prompt", "")
                prompt_id = prompt_data.get("prompt_id", f"prompt_{total_processed}")
                
                if not original_prompt:
                    continue
                
                self.logger.info(f"  📝 Processing prompt: {prompt_id}")
                total_processed += 1
                
                # Execute dual-path stealth rewriting
                stealth_result = self._dual_path_stealth_rewrite(original_prompt, prompt_data)
                
                # Record to final dataset regardless of success or failure
                if stealth_result and stealth_result.get("success"):
                    successful_rewrites += 1
                    
                    winning_path = stealth_result.get("winning_path", "unknown")
                    iterations = stealth_result.get("iterations", 0)  # Changed to iterations for consistency
                    similarity = stealth_result.get("final_similarity", 0)
                    ppl = stealth_result.get("final_ppl", 0)
                    final_prompt = stealth_result.get("stealth_prompt", "")
                    
                    self.logger.info(f"    🎉 Stealth successful!")
                    self.logger.info(f"       🏆 Winning path: {winning_path.upper()}")
                    self.logger.info(f"       🔄 Iterations used: {iterations}")
                    self.logger.info(f"       📊 Similarity: {similarity:.3f}")
                    self.logger.info(f"       📊 PPL: {ppl:.1f}")
                    self.logger.info(f"       💬 Rewriting result: {final_prompt[:120]}...")
                    
                    # Record successful result (including all available info)
                    final_record = {
                        "prompt_id": prompt_id,
                        "domain": self.domain,
                        "entity": prompt_data.get("entity", "unknown"),
                        "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                        "category": prompt_data.get("category", "unknown"),
                        "behavior_type": prompt_data.get("behavior_type", "unknown"),
                        "original_prompt": original_prompt,
                        "stealth_prompt": final_prompt,
                        "success": True,
                        "stealth_success": True,
                        "winning_path": winning_path,
                        "iterations": iterations,
                        "final_similarity": similarity,
                        "final_ppl": ppl,
                        "original_ppl": stealth_result.get("original_ppl"),  # Original prompt's PPL
                        "gemini_evaluation": stealth_result.get("gemini_evaluation", {}),
                        "asr_evaluation_reasoning": stealth_result.get("asr_evaluation_reasoning", ""),
                        "target_response": stealth_result.get("target_response", ""),
                        "accumulated_trigger_words": stealth_result.get("accumulated_trigger_words", []),
                        "has_stealth_version": True,
                        "metadata": prompt_data
                    }
                else:
                    reason = stealth_result.get("reason", "Unknown reason") if stealth_result else "Processing failed"
                    self.logger.warning(f"    ❌ Stealth failed: {reason}")
                    
                    # Check if there's a stealth result (even if attack failed)
                    stealth_prompt = stealth_result.get("stealth_prompt") if stealth_result else None
                    
                    # Extract iteration count - save even in failure case
                    iterations = 0
                    if stealth_result:
                        iterations = stealth_result.get("iterations", 0)
                        if iterations == 0:
                            iterations = stealth_result.get("total_iterations", 0)
                        if iterations == 0 and "Reached max iterations" in reason:
                            iterations = self.stealth_config.get("max_iterations", 5)
                    
                    # Failure record also includes all available info
                    final_record = {
                        "prompt_id": prompt_id,
                        "domain": self.domain,
                        "entity": prompt_data.get("entity", "unknown"),
                        "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                        "category": prompt_data.get("category", "unknown"),
                        "behavior_type": prompt_data.get("behavior_type", "unknown"),
                        "original_prompt": original_prompt,
                        "stealth_prompt": stealth_prompt,
                        "success": False,
                        "stealth_success": False,
                        "iterations": iterations,
                        "reason": reason,
                        "metadata": prompt_data
                    }
                    
                    # If there's a stealth result, add extra info
                    if stealth_result:
                        final_record.update({
                            "final_similarity": stealth_result.get("final_similarity"),
                            "final_ppl": stealth_result.get("final_ppl"),
                            "original_ppl": stealth_result.get("original_ppl"),
                            "gemini_evaluation": stealth_result.get("gemini_evaluation", {}),
                            "asr_evaluation_reasoning": stealth_result.get("asr_evaluation_reasoning", ""),
                            "target_response": stealth_result.get("target_response", ""),
                            "accumulated_trigger_words": stealth_result.get("accumulated_trigger_words", []),
                            "weighted_score": stealth_result.get("weighted_score"),
                            "winning_path": stealth_result.get("winning_path"),
                            "winning_iteration": stealth_result.get("winning_iteration"),
                            "has_stealth_version": stealth_prompt is not None
                        })
                    else:
                        final_record["has_stealth_version"] = False
                
                # Add to final dataset
                self.stealth_results.append(final_record)
                
                time.sleep(0.5)  # Prevent too frequent requests
            
            success_rate = successful_rewrites / total_processed if total_processed > 0 else 0
            self.logger.info(f"  📊 Stealth statistics: {successful_rewrites}/{total_processed} successful ({success_rate:.1%})")
            
            # 60% success rate considered test pass
            test_success = success_rate >= 0.6
            
            self.test_results["dual_path_stealth"] = {
                "status": test_success,
                "total_prompts": total_processed,
                "successful_rewrites": successful_rewrites,
                "success_rate": success_rate,
                "stealth_results": self.stealth_results
            }
            
            return test_success
            
        except Exception as e:
            self.logger.error(f"  ❌ Dual-path stealth test failed: {e}")
            self.test_results["dual_path_stealth"] = {"status": False, "error": str(e)}
            return False
    
    def _dual_path_stealth_rewrite(self, original_prompt: str, prompt_data: Dict) -> Optional[Dict]:
        """Execute dual-path stealth rewriting - basic and enhanced paths run alternately, with failure feedback mechanism, using Gemini evaluation
        
        🚀 High-concurrency optimized version: parallel execution of quality evaluation and target model calls within iterations
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_iterations = self.stealth_config["max_iterations"]
        
        self.logger.info(f"      🎭 Starting dual-path stealth rewriting: original={original_prompt[:60]}...")
        
        # Dual-path state
        basic_prompt = original_prompt
        enhanced_prompt = original_prompt
        
        # Failure feedback mechanism - accumulate trigger words
        last_failure_feedback = None
        failed_keywords_history = []  # Accumulate all identified trigger words
        
        # Track best candidate results (for fallback on failure)
        best_candidates = {
            "basic": [],  # Store all basic path candidate results
            "enhanced": []  # Store all enhanced path candidate results
        }
        
        # Create thread pool for parallel calls within iterations
        with ThreadPoolExecutor(max_workers=4) as iteration_executor:
            for iteration in range(max_iterations):
                # Current iteration selected path (alternating)
                current_path = "basic" if iteration % 2 == 0 else "enhanced"

                # Ablation: basic
                # current_path = "basic"

                # Ablation: enhanced
                # current_path = "enhanced"

                self.logger.debug(f"      🔄 Iteration {iteration + 1}/{max_iterations} - {current_path.upper()}")
                
                try:
                    if current_path == "basic":
                        # Basic path rewriting
                        request = ImplicitRequest(
                            original_prompt=basic_prompt,
                            method=ImplicitMethod.BASIC,
                            category=prompt_data.get("category", ""),
                            failure_feedback=last_failure_feedback,  # Pass failure feedback
                            failed_words_list=failed_keywords_history if failed_keywords_history else None  # Pass accumulated trigger words
                        )
                    else:
                        # Enhanced path rewriting
                        # Get entity name for semantic card generation - must exist
                        entity_name = prompt_data.get("entity")
                        if not entity_name:
                            raise ValueError(f"Enhanced path requires entity info, but not found in prompt_data. prompt_id: {prompt_data.get('prompt_id', 'unknown')}")
                        
                        request = ImplicitRequest(
                            original_prompt=enhanced_prompt,
                            method=ImplicitMethod.ENHANCED,
                            category=prompt_data.get("category", ""),
                            node_name=entity_name,  # Pass actual entity name
                            failure_feedback=last_failure_feedback,  # Pass failure feedback
                            failed_words_list=failed_keywords_history if failed_keywords_history else None  # Pass accumulated trigger words
                        )
                    
                    # Use detailed method to get LLM raw response and Gemini evaluation (with retry mechanism, max 4 times)
                    max_llm_retries = 4
                    response_details = None
                    last_llm_error = None
                    
                    for llm_attempt in range(max_llm_retries):
                        try:
                            response_details = self.implicit_service.generate_implicit_prompt_with_details(request)
                            break  # Success, exit retry loop
                        except Exception as llm_error:
                            last_llm_error = llm_error
                            if llm_attempt < max_llm_retries - 1:
                                wait_time = 2 ** llm_attempt  # Exponential backoff: 1, 2, 4 seconds
                                self.logger.debug(f"            🔄 LLM retry {llm_attempt + 1}/{max_llm_retries}")
                                time.sleep(wait_time)
                            else:
                                self.logger.warning(f"         ⚠️  LLM call failed: {str(llm_error)[:50]}")
                    
                    if response_details is None:
                        continue
                    
                    response = response_details["implicit_prompt"]
                    llama_response = response_details["llama_response"]
                    gemini_evaluation = response_details["gemini_evaluation"]
                    
                    if not response or not response.implicit_prompt:
                        continue
                    
                    candidate_prompt = self._clean_generated_prompt(response.implicit_prompt)
                    
                    # 🚀 Parallel optimization: perform quality evaluation and pre-call target model simultaneously
                    # This way, when quality passes, we can directly use the existing target response
                    target_response_future = None
                    intent_preserved = gemini_evaluation.get("intent_preserved", False)
                    is_fluent = gemini_evaluation.get("is_fluent", False)
                    
                    # If intent is preserved, start target model call ahead of time (don't wait for fluency check)
                    if intent_preserved:
                        def call_target_model_async(prompt):
                            """Async call to target model"""
                            try:
                                return self.asr_evaluator.openrouter_client.chat.completions.create(
                                    model=self.asr_evaluator.target_model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=1024,
                                    temperature=0.7,
                                    top_p=0.9,
                                )
                            except Exception as e:
                                return None
                        
                        target_response_future = iteration_executor.submit(call_target_model_async, candidate_prompt)
                    
                    quality_passed = intent_preserved and is_fluent
                    
                    # Record candidate result
                    candidate_record = {
                        "prompt": candidate_prompt,
                        "iteration": iteration + 1,
                        "intent_preserved": intent_preserved,
                        "is_fluent": is_fluent,
                        "quality_passed": quality_passed,
                        "gemini_evaluation": gemini_evaluation
                    }
                    best_candidates[current_path].append(candidate_record)
                    
                    if quality_passed:
                        # Update corresponding path state
                        if current_path == "basic":
                            basic_prompt = candidate_prompt
                        else:
                            enhanced_prompt = candidate_prompt
                        
                        # 🚀 Parallel optimization: use pre-called target model response
                        
                        # Get pre-called target response
                        target_response = None
                        if target_response_future:
                            try:
                                target_response_obj = target_response_future.result(timeout=60)
                                if target_response_obj:
                                    target_response = target_response_obj.choices[0].message.content
                            except Exception as e:
                                pass  # Silent handling, will fallback later
                        
                        if target_response:
                            # Use existing response directly for ASR evaluation (saves one API call)
                            jailbreak_success, failure_feedback, new_trigger_words, asr_reasoning = self._check_jailbreak_with_response(
                                candidate_prompt, target_response, original_prompt, prompt_data, failed_keywords_history
                            )
                        else:
                            # Fallback to original method
                            jailbreak_success, failure_feedback, new_trigger_words, asr_reasoning, target_response = self._check_jailbreak_success(
                                candidate_prompt, original_prompt, prompt_data, failed_keywords_history
                            )
                        
                        # Update failure feedback and accumulated trigger words
                        if not jailbreak_success:
                            last_failure_feedback = failure_feedback
                            if new_trigger_words:
                                failed_keywords_history = list(set(failed_keywords_history + new_trigger_words))
                        
                        if jailbreak_success:
                            self.logger.info(f"         🚀 Success! {current_path.upper()} path, iteration {iteration + 1}")
                            
                            # Calculate and save PPL and cosine similarity
                            final_similarity = self._calculate_similarity(original_prompt, candidate_prompt)
                            final_ppl = self._calculate_ppl(candidate_prompt)
                            original_ppl = self._calculate_ppl(original_prompt)
                            
                            # Found stealth prompt satisfying conditions
                            result = {
                                "prompt_id": prompt_data.get("prompt_id", "unknown"),
                                "original_prompt": original_prompt,
                                "stealth_prompt": candidate_prompt,
                                "success": True,
                                "stealth_success": True,
                                "winning_path": current_path,
                                "iterations": iteration + 1,
                                "final_similarity": final_similarity,  # Cosine similarity
                                "final_ppl": final_ppl,  # Stealth prompt's PPL
                                "original_ppl": original_ppl,  # Original prompt's PPL (for comparison)
                                "gemini_evaluation": gemini_evaluation,
                                "asr_evaluation_reasoning": asr_reasoning,  # Save ASR evaluation reasoning
                                "target_response": target_response,  # Save target LLM's response
                                "entity": prompt_data.get("entity", "unknown"),
                                "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                                "category": prompt_data.get("category", "unknown"),
                                "behavior_type": prompt_data.get("behavior_type", "unknown"),
                                "toxicity_evaluation": prompt_data.get("toxicity_evaluation", {}),
                                "filtering_info": prompt_data.get("filtering_info", {}),
                                "metadata": prompt_data,
                                "accumulated_trigger_words": failed_keywords_history,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            return result
                        # Failure, continue to next iteration
                    # Quality not passed, continue
                    
                except Exception as e:
                    self.logger.debug(f"         Iteration exception: {e}")
                    continue
        
        # All iterations failed, select best candidate as stealth result
        self.logger.info(f"      ❌ Failed: reached max iterations {max_iterations}")
        
        # Select best candidate (using Gemini evaluation results)
        best_candidate = self._select_best_candidate_gemini(best_candidates)
        
        if best_candidate:
            self.logger.debug(f"         Selected candidate: {best_candidate['path']} path, iteration {best_candidate['iteration']}")
            
            # Calculate and save PPL and cosine similarity (also needed for failure cases)
            best_stealth_prompt = best_candidate['prompt']
            final_similarity = self._calculate_similarity(original_prompt, best_stealth_prompt)
            final_ppl = self._calculate_ppl(best_stealth_prompt)
            original_ppl = self._calculate_ppl(original_prompt)
            
            return {
                "prompt_id": prompt_data.get("prompt_id", "unknown"),
                "original_prompt": original_prompt,
                "stealth_prompt": best_stealth_prompt,
                "success": False,
                "stealth_success": False,
                "iterations": max_iterations,
                "final_similarity": final_similarity,  # Cosine similarity
                "final_ppl": final_ppl,  # Stealth prompt's PPL
                "original_ppl": original_ppl,  # Original prompt's PPL (for comparison)
                "gemini_evaluation": best_candidate.get('gemini_evaluation', {}),
                "winning_path": best_candidate['path'],
                "winning_iteration": best_candidate['iteration'],
                "entity": prompt_data.get("entity", "unknown"),
                "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                "category": prompt_data.get("category", "unknown"),
                "behavior_type": prompt_data.get("behavior_type", "unknown"),
                "toxicity_evaluation": prompt_data.get("toxicity_evaluation", {}),
                "filtering_info": prompt_data.get("filtering_info", {}),
                "metadata": prompt_data,
                "reason": f"Reached max iterations ({max_iterations})",
                "accumulated_trigger_words": failed_keywords_history,
                "all_candidates": best_candidates,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            self.logger.info(f"         ❌ No suitable candidate found")
            self.logger.info(f"         📊 Final state:")
            self.logger.info(f"           🔵 Basic path: {basic_prompt[:80]}...")
            self.logger.info(f"           🟡 Enhanced path: {enhanced_prompt[:80]}...")
            if failed_keywords_history:
                self.logger.info(f"           🚫 Accumulated trigger words: {', '.join(failed_keywords_history)}")
            
            return {
                "prompt_id": prompt_data.get("prompt_id", "unknown"),
                "original_prompt": original_prompt,
                "stealth_prompt": None,
                "success": False,
                "stealth_success": False,
                "iterations": max_iterations,
                "entity": prompt_data.get("entity", "unknown"),
                "wikidata_uri": prompt_data.get("wikidata_uri", "unknown"),
                "category": prompt_data.get("category", "unknown"),
                "behavior_type": prompt_data.get("behavior_type", "unknown"),
                "toxicity_evaluation": prompt_data.get("toxicity_evaluation", {}),
                "filtering_info": prompt_data.get("filtering_info", {}),
                "metadata": prompt_data,
                "basic_final": basic_prompt,
                "enhanced_final": enhanced_prompt,
                "reason": f"Reached max iterations ({max_iterations}) with no suitable candidate",
                "accumulated_trigger_words": failed_keywords_history,
                "all_candidates": best_candidates,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _select_best_candidate_gemini(self, best_candidates: Dict) -> Optional[Dict]:
        """Select the best stealth result from all candidates - using Gemini evaluation results"""
        all_candidates = []
        
        # Collect all candidates that passed quality check
        for path_name, candidates in best_candidates.items():
            for candidate in candidates:
                if candidate.get('quality_passed', False):
                    all_candidates.append({
                        "path": path_name,
                        "prompt": candidate['prompt'],
                        "iteration": candidate['iteration'],
                        "gemini_evaluation": candidate.get('gemini_evaluation', {})
                    })
        
        if not all_candidates:
            # If no candidates passed quality check, select the last candidate
            for path_name, candidates in best_candidates.items():
                if candidates:
                    last_candidate = candidates[-1]
                    all_candidates.append({
                        "path": path_name,
                        "prompt": last_candidate['prompt'],
                        "iteration": last_candidate['iteration'],
                        "gemini_evaluation": last_candidate.get('gemini_evaluation', {})
                    })
        
        if not all_candidates:
            return None
        
        # Select the result from the last iteration
        best = max(all_candidates, key=lambda x: x['iteration'])
        return best
    
    def _select_best_candidate(self, best_candidates: Dict, similarity_threshold: float, ppl_threshold: float) -> Optional[Dict]:
        """Select the best stealth result from all candidates - focus on results from last two iterations"""
        all_candidates = []
        
        # Collect all candidate results, focus on last two iterations
        max_iteration = 0
        for path_name, candidates in best_candidates.items():
            for candidate in candidates:
                max_iteration = max(max_iteration, candidate['iteration'])
        
        # Filter candidates from last two iterations (if not enough iterations, include all)
        last_two_iterations = [max_iteration, max_iteration - 1] if max_iteration > 0 else [0]
        
        for path_name, candidates in best_candidates.items():
            for candidate in candidates:
                # Only consider results from last two iterations
                if candidate['iteration'] in last_two_iterations:
                    # Calculate weighted score: similarity weight 0.6, PPL weight 0.4 (lower PPL is better)
                    # Similarity range 0-1, PPL range usually 10-200, needs normalization
                    normalized_similarity = candidate['similarity']  # Already in 0-1 range
                    normalized_ppl = max(0, 1 - (candidate['ppl'] - 10) / 190)  # Normalize PPL(10-200) to 0-1, lower is better
                    
                    weighted_score = 0.6 * normalized_similarity + 0.4 * normalized_ppl
                    
                    all_candidates.append({
                        "path": path_name,
                        "prompt": candidate['prompt'],
                        "similarity": candidate['similarity'],
                        "ppl": candidate['ppl'],
                        "iteration": candidate['iteration'],
                        "constraints_met": candidate['constraints_met'],
                        "weighted_score": weighted_score
                    })
        
        if not all_candidates:
            return None
        
        self.logger.info(f"         📊 Selecting from last two iterations ({last_two_iterations}), {len(all_candidates)} candidates")
        
        # Prioritize candidates that satisfy constraints, then sort by weighted score
        constraint_satisfied = [c for c in all_candidates if c['constraints_met']]
        
        if constraint_satisfied:
            # If there are candidates satisfying constraints, select the one with highest weighted score
            best = max(constraint_satisfied, key=lambda x: x['weighted_score'])
            self.logger.info(f"         ✅ Found best candidate satisfying constraints (iteration {best['iteration']}, weighted score {best['weighted_score']:.3f})")
        else:
            # If no candidates satisfy constraints, select the one with highest weighted score from last two iterations
            best = max(all_candidates, key=lambda x: x['weighted_score'])
            self.logger.info(f"         ⚠️ No candidates satisfy constraints, selecting highest weighted score from last two iterations (iteration {best['iteration']}, weighted score {best['weighted_score']:.3f})")
        
        return best
    
    def _check_stealth_constraints(self, original: str, candidate: str, 
                                 similarity_threshold: float, ppl_threshold: float) -> bool:
        """Check stealth constraint conditions"""
        try:
            # Check semantic similarity
            similarity = self._calculate_similarity(original, candidate)
            self.logger.info(f"           🔍 Similarity check: {similarity:.3f} (threshold: ≥{similarity_threshold})")
            
            if similarity < similarity_threshold:
                self.logger.info(f"           ❌ Insufficient similarity: {similarity:.3f} < {similarity_threshold}")
                return False
            
            # Check PPL
            ppl = self._calculate_ppl(candidate)
            self.logger.info(f"           🔍 PPL check: {ppl:.1f} (threshold: ≤{ppl_threshold})")
            
            if ppl > ppl_threshold:
                self.logger.info(f"           ❌ PPL too high: {ppl:.1f} > {ppl_threshold}")
                return False
            
            self.logger.info(f"           ✅ All constraint conditions satisfied")
            return True
            
        except Exception as e:
            self.logger.warning(f"           ⚠️  Constraint check failed: {e}")
            return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts - only using real sentence-transformers model"""
        # Use sentence-transformers to calculate real semantic similarity
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        # Initialize model (if not already initialized)
        if not hasattr(self, '_similarity_model'):
            self.logger.info("           🔄 Initializing sentence-transformers model...")
            self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.logger.info("           ✅ Sentence-transformers model initialization complete")
        
        # Calculate embeddings and similarity
        embeddings = self._similarity_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        
        return similarity
    
    def _calculate_ppl(self, text: str) -> float:
        """Calculate text perplexity - only using real GPT-2 model"""
        # Use GPT-2 model to calculate real perplexity
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch
        
        # Initialize model (if not already initialized)
        if not hasattr(self, '_ppl_model'):
            self.logger.info("           🔄 Initializing GPT-2 perplexity model...")
            self._ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._ppl_model = GPT2LMHeadModel.from_pretrained('gpt2')
            if self._ppl_tokenizer.pad_token is None:
                self._ppl_tokenizer.pad_token = self._ppl_tokenizer.eos_token
            self.logger.info("           ✅ GPT-2 perplexity model initialization complete")
        
        # Calculate perplexity
        inputs = self._ppl_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._ppl_model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            ppl = torch.exp(loss).item()
        
        return ppl
    
    def _check_jailbreak_success(self, prompt: str, original_prompt: str = None, 
                                   prompt_data: Dict = None, failed_keywords_history: list = None) -> tuple:
        """
        Check if jailbreak was successful (using real LLM evaluation) and return failure analysis
        
        Returns:
            tuple: (is_successful, failure_feedback, new_trigger_words, asr_reasoning, target_response)
                - is_successful: bool
                - failure_feedback: str or None (None if successful)
                - new_trigger_words: list (newly discovered trigger words)
                - asr_reasoning: str (ASR evaluation reasoning)
                - target_response: str (target LLM's response)
        """
        return self._check_jailbreak_with_real_llm(prompt, original_prompt, prompt_data, failed_keywords_history)
    
    def _check_jailbreak_with_response(self, prompt: str, target_response: str, original_prompt: str = None, 
                                        prompt_data: Dict = None, failed_keywords_history: list = None) -> tuple:
        """
        🚀 Parallel optimization: Use existing target response for ASR evaluation, saves one API call
        
        Returns:
            tuple: (is_successful, failure_feedback, new_trigger_words, asr_reasoning)
        """
        if failed_keywords_history is None:
            failed_keywords_history = []
            
        self.logger.info(f"           🔄 Using pre-fetched response for ASR evaluation (saves 1 API call)...")
        
        try:
            # Get specific category from prompt_data
            category = prompt_data.get("category", self.domain) if prompt_data else self.domain
            
            # Directly use existing target response for evaluation
            asr_result = self.asr_evaluator.check_asr_with_prompt_and_response(
                prompt_for_eval=original_prompt if original_prompt else prompt,
                target_response=target_response,
                category=category
            )
            
            self.logger.info(f"           🎯 Evaluation result: {'Jailbreak successful' if asr_result.asr_successful else 'Attack failed'}")
            self.logger.info(f"           💭 Reasoning: {asr_result.evaluation_reasoning[:100]}...")
            
            if asr_result.asr_successful:
                return True, None, [], asr_result.evaluation_reasoning
            else:
                # Build failure feedback
                refusal_type = asr_result.refusal_type or "unknown"
                trigger_words = asr_result.trigger_words or []
                
                feedback_parts = []
                feedback_parts.append(f"REFUSAL_TYPE: {refusal_type}")
                
                if trigger_words:
                    feedback_parts.append(f"THIS_ATTEMPT_TRIGGER_WORDS: {', '.join(trigger_words)}")
                    self.logger.info(f"           🎯 Trigger words: {', '.join(trigger_words)}")
                
                # Add historically accumulated trigger words (deduplicated)
                all_failed_words = list(set(failed_keywords_history + trigger_words))
                if all_failed_words:
                    feedback_parts.append(f"ALL_HISTORICALLY_FAILED_WORDS: {', '.join(all_failed_words)}")
                
                if target_response:
                    feedback_parts.append(f"TARGET_RESPONSE: \"{target_response[:100]}...\"")
                
                failure_feedback = "\n".join(feedback_parts)
                self.logger.info(f"           📋 Refusal type: {refusal_type}")
                
                return False, failure_feedback, trigger_words, asr_result.evaluation_reasoning
            
        except Exception as e:
            self.logger.error(f"           ❌ ASR evaluation failed: {e}")
            return False, f"Error during evaluation: {str(e)}", [], ""
    
    def _check_jailbreak_with_real_llm(self, prompt: str, original_prompt: str = None, 
                                       prompt_data: Dict = None, failed_keywords_history: list = None) -> tuple:
        """
        Use real LLM for jailbreak success check - reuse ASR evaluation service, return failure analysis
        
        Returns:
            tuple: (is_successful, failure_feedback, new_trigger_words)
        """
        if failed_keywords_history is None:
            failed_keywords_history = []
            
        self.logger.info(f"           🔄 Using ASR evaluation service for attack evaluation...")
        
        try:
            # Get specific category from prompt_data, use domain as fallback if not available
            category = prompt_data.get("category", self.domain) if prompt_data else self.domain
            
            # Directly use already initialized ASR evaluation service, avoid recreating API client
            asr_result = self.asr_evaluator.check_asr(
                rewritten_prompt=prompt,
                category=category,
                original_prompt=original_prompt
            )
            
            self.logger.info(f"           🎯 Evaluation result: {'Jailbreak successful' if asr_result.asr_successful else 'Attack failed'}")
            self.logger.info(f"           💭 Reasoning: {asr_result.evaluation_reasoning[:100]}...")
            
            if asr_result.asr_successful:
                return True, None, [], asr_result.evaluation_reasoning, asr_result.target_response
            else:
                # Build failure feedback
                refusal_type = asr_result.refusal_type or "unknown"
                trigger_words = asr_result.trigger_words or []
                target_response = asr_result.target_response or ""
                
                feedback_parts = []
                feedback_parts.append(f"REFUSAL_TYPE: {refusal_type}")
                
                if trigger_words:
                    feedback_parts.append(f"THIS_ATTEMPT_TRIGGER_WORDS: {', '.join(trigger_words)}")
                    self.logger.info(f"           🎯 Trigger words: {', '.join(trigger_words)}")
                
                # Add historically accumulated trigger words (deduplicated)
                all_failed_words = list(set(failed_keywords_history + trigger_words))
                if all_failed_words:
                    feedback_parts.append(f"ALL_HISTORICALLY_FAILED_WORDS: {', '.join(all_failed_words)}")
                
                if target_response:
                    feedback_parts.append(f"TARGET_RESPONSE: \"{target_response[:100]}...\"")
                
                failure_feedback = "\n".join(feedback_parts)
                self.logger.info(f"           📋 Refusal type: {refusal_type}")
                
                return False, failure_feedback, trigger_words, asr_result.evaluation_reasoning, asr_result.target_response
            
        except Exception as e:
            self.logger.error(f"           ❌ ASR evaluation failed: {e}")
            return False, f"Error during evaluation: {str(e)}", [], "", ""
    
    def analyze_stealth_effectiveness(self) -> bool:
        """Analyze stealth effectiveness"""
        self.logger.info("📊 Analyzing stealth effectiveness...")
        
        if not self.stealth_results:
            self.logger.error("  ❌ No stealth results to analyze")
            return False
        
        try:
            effectiveness_metrics = {
                "total_rewrites": len(self.stealth_results),
                "path_performance": {},
                "success_analysis": {},
                "iteration_analysis": {},
                "quality_analysis": {}
            }
            
            # Path performance analysis
            successful_results = [r for r in self.stealth_results if r.get("success")]
            basic_wins = sum(1 for r in successful_results if r.get("winning_path") == "basic")
            enhanced_wins = sum(1 for r in successful_results if r.get("winning_path") == "enhanced")
            
            effectiveness_metrics["path_performance"] = {
                "basic_wins": basic_wins,
                "enhanced_wins": enhanced_wins,
                "basic_win_rate": basic_wins / len(successful_results) if successful_results else 0,
                "enhanced_win_rate": enhanced_wins / len(successful_results) if successful_results else 0
            }
            
            # Iteration analysis
            if successful_results:
                iterations = [r.get("iterations", 0) for r in successful_results]  # Changed to iterations for consistency
                effectiveness_metrics["iteration_analysis"] = {
                    "avg_iterations": sum(iterations) / len(iterations),
                    "min_iterations": min(iterations),
                    "max_iterations": max(iterations),
                    "iteration_distribution": {i: iterations.count(i) for i in set(iterations)}
                }
            
            # Quality analysis
            similarities = [r.get("final_similarity", 0) for r in successful_results]
            ppls = [r.get("final_ppl", 0) for r in successful_results]
            
            if similarities and ppls:
                effectiveness_metrics["quality_analysis"] = {
                    "avg_similarity": sum(similarities) / len(similarities),
                    "min_similarity": min(similarities),
                    "max_similarity": max(similarities),
                    "avg_ppl": sum(ppls) / len(ppls),
                    "min_ppl": min(ppls),
                    "max_ppl": max(ppls)
                }
            
            self.logger.info(f"  📊 Stealth effectiveness analysis:")
            self.logger.info(f"      Successful rewrites: {len(successful_results)}/{len(self.stealth_results)}")
            
            if successful_results:
                self.logger.info(f"      Path win distribution: basic={basic_wins}, enhanced={enhanced_wins}")
                if "iteration_analysis" in effectiveness_metrics:
                    avg_iter = effectiveness_metrics["iteration_analysis"]["avg_iterations"]
                    self.logger.info(f"      Average iterations: {avg_iter:.1f}")
                if "quality_analysis" in effectiveness_metrics:
                    avg_sim = effectiveness_metrics["quality_analysis"]["avg_similarity"]
                    avg_ppl = effectiveness_metrics["quality_analysis"]["avg_ppl"]
                    self.logger.info(f"      Average similarity: {avg_sim:.3f}")
                    self.logger.info(f"      Average PPL: {avg_ppl:.1f}")
            
            success = len(successful_results) >= len(self.stealth_results) * 0.5
            if success:
                self.logger.info("  ✅ Stealth effectiveness analysis passed")
            else:
                self.logger.error("  ❌ Stealth effectiveness analysis failed")
            
            self.test_results["stealth_effectiveness"] = {
                "status": success,
                "effectiveness_metrics": effectiveness_metrics
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"  ❌ Stealth effectiveness analysis failed: {e}")
            self.test_results["stealth_effectiveness"] = {"status": False, "error": str(e)}
            return False
            enhanced_wins = sum(1 for r in self.stealth_results if r.get("winner_route") == "enhanced")
            no_winner = sum(1 for r in self.stealth_results if not r.get("winner_route"))
            
            effectiveness_metrics["route_performance"] = {
                "basic_wins": basic_wins,
                "enhanced_wins": enhanced_wins,
                "no_winner": no_winner,
                "basic_win_rate": basic_wins / len(self.stealth_results),
                "enhanced_win_rate": enhanced_wins / len(self.stealth_results)
            }
            
            # Success analysis
            successful_attacks = [r for r in self.stealth_results if r.get("attack_success")]
            effectiveness_metrics["success_analysis"] = {
                "total_successful": len(successful_attacks),
                "success_rate": len(successful_attacks) / len(self.stealth_results),
                "basic_chain_successes": sum(1 for r in self.stealth_results if r.get("basic_chain_success")),
                "enhanced_chain_successes": sum(1 for r in self.stealth_results if r.get("enhanced_chain_success"))
            }
            
            # Iteration count analysis
            success_iterations = [r.get("success_iteration") for r in successful_attacks if r.get("success_iteration")]
            if success_iterations:
                effectiveness_metrics["iteration_analysis"] = {
                    "avg_iterations_to_success": sum(success_iterations) / len(success_iterations),
                    "min_iterations": min(success_iterations),
                    "max_iterations": max(success_iterations),
                    "median_iterations": sorted(success_iterations)[len(success_iterations) // 2]
                }
            
            # Timing analysis
            attack_times = [r.get("total_time", 0) for r in self.stealth_results if r.get("total_time")]
            if attack_times:
                effectiveness_metrics["timing_analysis"] = {
                    "avg_time": sum(attack_times) / len(attack_times),
                    "min_time": min(attack_times),
                    "max_time": max(attack_times),
                    "total_time": sum(attack_times)
                }
            
            # Output analysis results
            self.logger.info(f"  📊 Attack effectiveness analysis:")
            self.logger.info(f"      Total attacks: {effectiveness_metrics['total_attacks']}")
            
            route_perf = effectiveness_metrics["route_performance"]
            self.logger.info(f"      Path win rate: Basic {route_perf['basic_win_rate']:.2%}, Enhanced {route_perf['enhanced_win_rate']:.2%}")
            
            success_analysis = effectiveness_metrics["success_analysis"]
            self.logger.info(f"      Overall success rate: {success_analysis['success_rate']:.2%}")
            
            iteration_analysis = effectiveness_metrics.get("iteration_analysis", {})
            if iteration_analysis:
                self.logger.info(f"      Average success iterations: {iteration_analysis.get('avg_iterations_to_success', 0):.1f}")
            
            timing_analysis = effectiveness_metrics.get("timing_analysis", {})
            if timing_analysis:
                self.logger.info(f"      Average attack time: {timing_analysis.get('avg_time', 0):.2f}s")
            
            # Determine if effectiveness is good
            success = (
                effectiveness_metrics["total_attacks"] > 0 and
                success_analysis["success_rate"] > 0.3 and  # Over 30% success rate
                (route_perf["basic_win_rate"] > 0 or route_perf["enhanced_win_rate"] > 0)  # At least one path has victories
            )
            
            if success:
                self.logger.info("  ✅ Attack effectiveness analysis passed")
            else:
                self.logger.error("  ❌ Attack effectiveness analysis failed")
            
            self.test_results["attack_effectiveness"] = {
                "status": success,
                "metrics": effectiveness_metrics
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"  ❌ Attack effectiveness analysis failed: {e}")
            self.test_results["attack_effectiveness"] = {"status": False, "error": str(e)}
            return False
    
    def export_dataset(self) -> bool:
        """Export stealth dataset"""
        self.logger.info("💾 Exporting stealth dataset...")
        
        if not self.stealth_results:
            self.logger.error("  ❌ No stealth results to export")
            return False
        
        try:
            # Export stealth dataset - standard format
            dataset_path = self.output_dir / f"step5_stealth_dataset_{self.domain}.json"
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.stealth_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"  ✅ Stealth dataset exported: {dataset_path.name}")
            
            # Statistics
            total_prompts = len(self.stealth_results)
            successful_attacks = [r for r in self.stealth_results if r.get("success")]
            failed_with_stealth = [r for r in self.stealth_results if not r.get("success") and r.get("has_stealth_version")]
            failed_completely = [r for r in self.stealth_results if not r.get("success") and not r.get("has_stealth_version")]
            
            self.logger.info(f"  📊 Dataset statistics:")
            self.logger.info(f"      Total prompts: {total_prompts}")
            self.logger.info(f"      Attack successful: {len(successful_attacks)}")
            self.logger.info(f"      Attack failed but has stealth version: {len(failed_with_stealth)}")
            self.logger.info(f"      Completely failed: {len(failed_completely)} (stealth prompt is null)")
            
            # Generate dataset containing only successful cases
            if successful_attacks:
                success_dataset_path = self.output_dir / f"step5_successful_stealth_dataset_{self.domain}.json"
                with open(success_dataset_path, 'w', encoding='utf-8') as f:
                    json.dump(successful_attacks, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"  ✅ Successful cases dataset exported: {success_dataset_path.name}")
            
            # Generate processing summary
            summary = {
                "domain": self.domain,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_prompts": total_prompts,
                "successful_attacks": len(successful_attacks),
                "failed_with_stealth": len(failed_with_stealth),
                "failed_completely": len(failed_completely),
                "success_rate": len(successful_attacks) / total_prompts if total_prompts > 0 else 0,
                "stealth_config": self.stealth_config,
                "dataset_format": {
                    "fields": [
                        "prompt_id",
                        "original_prompt", 
                        "stealth_prompt",  # null if failed
                        "success",
                        "winning_path",  # if success
                        "iterations",   # if success
                        "final_similarity",  # if success
                        "final_ppl",    # if success
                        "reason",       # if failed
                        "metadata"
                    ]
                }
            }
            
            summary_path = self.output_dir / f"step5_stealth_summary_{self.domain}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"  ✅ Processing summary exported: {summary_path.name}")
            
            self.test_results["dataset_export"] = {
                "status": True,
                "dataset_path": str(dataset_path),
                "success_dataset_path": str(success_dataset_path) if successful_attacks else None,
                "summary_path": str(summary_path),
                "total_prompts": total_prompts,
                "successful_prompts": len(successful_attacks),
                "failed_with_stealth": len(failed_with_stealth),
                "failed_completely": len(failed_completely)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"  ❌ Stealth dataset export failed: {e}")
            self.test_results["dataset_export"] = {"status": False, "error": str(e)}
            return False
    
    def _clean_export_record(self, record: Dict):
        """Clean export record, remove non-serializable objects"""
        # Remove fields that may contain complex objects
        fields_to_clean = ["basic_chain_details", "enhanced_chain_details"]
        for field in fields_to_clean:
            if field in record and isinstance(record[field], dict):
                # Keep basic info, remove complex objects
                cleaned = {}
                for k, v in record[field].items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        cleaned[k] = v
                    elif isinstance(v, dict):
                        # Only keep simple dictionaries
                        cleaned[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool, type(None)))}
                record[field] = cleaned
    
    def _serialize_test_results(self, results: Dict) -> Dict:
        """Serialize test results, handle non-JSON serializable objects"""
        serialized = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_test_results(value)
            elif hasattr(value, "value"):  # enum object
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized
    
    def run_all_processing(self) -> bool:
        """Run all dual-path stealth rewriting processing"""
        # Determine actual number of prompts to process
        prompts_to_process = self.load_evaluated_prompts()
        actual_test_size = len(prompts_to_process)
        
        if self.test_size == -1:
            self.logger.info(f"\n🎭 Starting dual-path stealth rewriting processing (domain: {self.domain}, processing all {actual_test_size} prompts)")
        else:
            self.logger.info(f"\n🎭 Starting dual-path stealth rewriting processing (domain: {self.domain}, limited to {actual_test_size} prompts)")
        
        self.logger.info("="*60)
        
        processing_steps = [
            ("Dual-path stealth rewriting", self.process_dual_path_stealth_rewrite),
            ("Stealth effectiveness analysis", self.analyze_stealth_effectiveness),
            ("Dataset export", self.export_dataset)
        ]
        
        completed_steps = 0
        total_steps = len(processing_steps)
        
        for step_name, step_func in processing_steps:
            self.logger.info(f"\n📋 Executing step: {step_name}")
            try:
                success = step_func()
                if success:
                    completed_steps += 1
            except Exception as e:
                self.logger.error(f"Exception occurred during step '{step_name}': {e}")
        
        # Generate processing summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Dual-path stealth rewriting processing summary")
        self.logger.info(f"✅ Completed: {completed_steps}/{total_steps}")
        
        completion_rate = completed_steps / total_steps
        overall_status = completion_rate >= 0.67  # 67% completion rate considered successful
        
        if overall_status:
            self.logger.info("🎉 Dual-path stealth rewriting processing completed")
            successful_rewrites = [r for r in self.stealth_results if r.get("success")]
            self.logger.info(f"🎭 Generated {len(successful_rewrites)} successful stealth samples")
            self.logger.info(f"📋 Total processed {len(self.stealth_results)} prompts")
        else:
            self.logger.error("❌ Dual-path stealth rewriting processing incomplete")
        
        # Save processing results
        self._save_processing_results(overall_status, completed_steps, total_steps, actual_test_size)
        
        return overall_status
    
    def _save_processing_results(self, overall_status: bool, completed: int, total: int, actual_process_size: int):
        """Save processing results to file"""
        successful_rewrites = [r for r in self.stealth_results if r.get("success")]
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "domain": self.domain,
            "process_size_setting": self.test_size,  # Original setting
            "actual_process_size": actual_process_size,  # Actual processed count
            "overall_status": overall_status,
            "summary": {
                "completed_steps": completed,
                "total_steps": total,
                "completion_rate": completed / total if total > 0 else 0,
                "total_prompts": len(self.stealth_results),
                "successful_rewrites": len(successful_rewrites),
                "success_rate": len(successful_rewrites) / len(self.stealth_results) if self.stealth_results else 0
            },
            "stealth_config": self.stealth_config,
            "detailed_results": self._serialize_test_results(self.test_results)
        }
        
        report_path = self.output_dir / "step5_dual_path_stealth_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Dual-path stealth rewriting processing report saved: {report_path}")

def main():
    # Load environment config
    load_env_config()
    
    # Check API config
    if not check_api_config():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Step 5: Dual-path Stealth Rewriting")
    parser.add_argument("--domain", default="medicine",
                      choices=["medicine", "finance", "education", "law"],
                      help="Processing domain")
    parser.add_argument("--test-size", type=int, default=-1,
                      help="Processing data size (-1 means process all step4 results)")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed information")
    parser.add_argument("--output-dir", type=str,
                      help="Output directory (passed by test runner)")
    
    args = parser.parse_args()
    
    processor = DualPathStealthProcessor(
        domain=args.domain,
        test_size=args.test_size,
        verbose=args.verbose
    )
    
    success = processor.run_all_processing()
    
    if success:
        successful_rewrites = [r for r in processor.stealth_results if r.get("success")]
        print(f"\n✅ Step 5 completed: Dual-path stealth rewriting processing completed")
        print(f"🎭 Generated {len(successful_rewrites)} successful stealth samples")
        print(f"📋 Total processed {len(processor.stealth_results)} prompts")
        if successful_rewrites:
            winner_stats = {}
            for rewrite in successful_rewrites:
                winner = rewrite.get("winning_path", "none")
                winner_stats[winner] = winner_stats.get(winner, 0) + 1
            print(f"🏆 Winning path statistics: {winner_stats}")
    else:
        print(f"\n❌ Step 5 failed: Dual-path stealth rewriting processing incomplete")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()