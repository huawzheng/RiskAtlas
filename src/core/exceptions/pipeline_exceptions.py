#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline-related Exception Definitions
"""

class PipelineError(Exception):
    """Pipeline Base Exception"""
    
    def __init__(self, message: str, stage: str = None, step: str = None):
        self.message = message
        self.stage = stage
        self.step = step
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format exception message"""
        parts = []
        if self.stage:
            parts.append(f"Stage[{self.stage}]")
        if self.step:
            parts.append(f"Step[{self.step}]")
        if parts:
            return f"{' '.join(parts)}: {self.message}"
        return self.message


class PipelineConfigError(PipelineError):
    """Pipeline Configuration Exception"""
    pass


class PipelineValidationError(PipelineError):
    """Pipeline Validation Exception"""
    pass


class PipelineExecutionError(PipelineError):
    """Pipeline Execution Exception"""
    
    def __init__(self, message: str, stage: str = None, step: str = None, original_error: Exception = None):
        super().__init__(message, stage, step)
        self.original_error = original_error


class StepExecutionError(PipelineExecutionError):
    """Step Execution Exception"""
    
    def __init__(self, step_name: str, message: str, original_error: Exception = None):
        super().__init__(message, step=step_name, original_error=original_error)
        self.step_name = step_name
