#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Seed Setting Utility
Ensures experiment reproducibility
"""

import random
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: Optional[int] = None, enable_deterministic: bool = True) -> int:
    """
    Set global random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed, uses default value 42 if None
        enable_deterministic: Whether to enable deterministic mode
        
    Returns:
        int: The actual random seed used
    """
    if seed is None:
        seed = 42
    
    # Set Python built-in random module seed
    random.seed(seed)
    
    # Set environment variable (if needed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set NumPy random seed (if available)
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"NumPy random seed set to: {seed}")
    except ImportError:
        logger.debug("NumPy not installed, skipping NumPy random seed setting")
    
    # Try to set PyTorch random seed (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if enable_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.debug(f"PyTorch random seed set to: {seed}")
    except ImportError:
        logger.debug("PyTorch not installed, skipping PyTorch random seed setting")
    
    logger.info(f"✅ Global random seed set to: {seed}")
    return seed


def load_seed_from_config(config_manager) -> int:
    """
    Load random seed setting from config manager
    
    Args:
        config_manager: Config manager instance
        
    Returns:
        int: Random seed value
    """
    try:
        reproducibility_config = config_manager.main_config.get('reproducibility', {})
        seed = reproducibility_config.get('random_seed', 42)
        enable_deterministic = reproducibility_config.get('enable_deterministic', True)
        
        return set_random_seed(seed, enable_deterministic)
    except Exception as e:
        logger.warning(f"Failed to load random seed from config file: {e}, using default seed 42")
        return set_random_seed(42, True)


def get_random_state_info() -> dict:
    """
    Get current random state info for debugging
    
    Returns:
        dict: Random state information
    """
    info = {
        "python_random_state": random.getstate()[1][0],  # Get first value of current state as identifier
        "pythonhashseed": os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    try:
        import numpy as np
        info["numpy_random_state"] = np.random.get_state()[1][0]
    except ImportError:
        info["numpy_random_state"] = "not_available"
    
    try:
        import torch
        info["torch_random_state"] = torch.initial_seed()
        info["torch_deterministic"] = torch.backends.cudnn.deterministic
    except ImportError:
        info["torch_random_state"] = "not_available"
        info["torch_deterministic"] = "not_available"
    
    return info