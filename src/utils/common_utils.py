#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Utility Functions
"""

import os
from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_directory(path):
    """Ensure directory exists (alias)"""
    return ensure_dir(path)
