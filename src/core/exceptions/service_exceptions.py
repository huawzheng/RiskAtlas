#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Service-related Exception Definitions
"""

class ServiceError(Exception):
    """Service Base Exception"""
    
    def __init__(self, message: str, service: str = None):
        self.message = message
        self.service = service
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format exception message"""
        if self.service:
            return f"Service[{self.service}]: {self.message}"
        return self.message


class KnowledgeGraphError(ServiceError):
    """Knowledge Graph Related Exception"""
    
    def __init__(self, message: str):
        super().__init__(message, "KnowledgeGraph")


class PromptGenerationError(ServiceError):
    """Prompt Generation Related Exception"""
    
    def __init__(self, message: str):
        super().__init__(message, "PromptGeneration")


class EvaluationError(ServiceError):
    """Evaluation Related Exception"""
    
    def __init__(self, message: str):
        super().__init__(message, "Evaluation")


class FilteringError(ServiceError):
    """Filtering Related Exception"""
    
    def __init__(self, message: str):
        super().__init__(message, "Filtering")


class DatabaseConnectionError(ServiceError):
    """Database Connection Exception"""
    
    def __init__(self, message: str, database_type: str = None):
        service_name = f"Database[{database_type}]" if database_type else "Database"
        super().__init__(message, service_name)


class ExportError(ServiceError):
    """Export Related Exception"""
    
    def __init__(self, message: str, export_type: str = None):
        service_name = f"Export[{export_type}]" if export_type else "Export"
        super().__init__(message, service_name)
