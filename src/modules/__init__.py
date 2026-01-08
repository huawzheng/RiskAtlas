#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Modules Package
Contains core functionalities for knowledge graph retrieval, harmful prompt generation, and toxicity evaluation
"""

# Import core modules
try:
    from .wikidata_retriever import WikidataRetriever
except ImportError:
    WikidataRetriever = None

try:
    from .harmful_prompt_generator import HarmfulPromptGenerator, HarmCategory, HarmPrompt
except ImportError:
    HarmfulPromptGenerator = None
    HarmCategory = None
    HarmPrompt = None

try:
    from .granite_guardian_evaluator import GraniteGuardianEvaluator, ToxicityResult
except ImportError:
    GraniteGuardianEvaluator = None
    ToxicityResult = None

# Export list
__all__ = []

# Dynamically add successfully imported modules
if WikidataRetriever is not None:
    __all__.append('WikidataRetriever')
if HarmfulPromptGenerator is not None:
    __all__.extend(['HarmfulPromptGenerator', 'HarmCategory', 'HarmPrompt'])
if GraniteGuardianEvaluator is not None:
    __all__.extend(['GraniteGuardianEvaluator', 'ToxicityResult'])
