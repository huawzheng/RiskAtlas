#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Operation Utility Module
Provides unified file read/write and management functionality
"""

import json
import csv
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_timestamped_filename(base_name: str, extension: str, include_date: bool = True) -> str:
    """
    Generate filename with timestamp
    
    Args:
        base_name: Base filename
        extension: File extension
        include_date: Whether to include date
        
    Returns:
        Filename with timestamp
    """
    if include_date:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%H%M%S")
    
    return f"{base_name}_{timestamp}.{extension}"

def save_json(data: Any, file_path: Union[str, Path], ensure_dir: bool = True) -> None:
    """
    Save data as JSON file
    
    Args:
        data: Data to save
        file_path: File path
        ensure_dir: Whether to ensure directory exists
    """
    file_path = Path(file_path)
    
    if ensure_dir:
        ensure_directory(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file
    
    Args:
        file_path: File path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_csv(data: Union[List[Dict], pd.DataFrame], file_path: Union[str, Path], ensure_dir: bool = True) -> None:
    """
    Save data as CSV file
    
    Args:
        data: Data to save (list of dicts or DataFrame)
        file_path: File path  
        ensure_dir: Whether to ensure directory exists
    """
    file_path = Path(file_path)
    
    if ensure_dir:
        ensure_directory(file_path.parent)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False, encoding='utf-8')
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
    else:
        raise ValueError("Data format not supported, please provide DataFrame or list of dicts")

def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path: File path
        
    Returns:
        DataFrame object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    return pd.read_csv(file_path, encoding='utf-8')

def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Backup file
    
    Args:
        file_path: Path of file to backup
        backup_dir: Backup directory, if None creates backup subdirectory in original directory
        
    Returns:
        Backup file path
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent / "backup"
    else:
        backup_dir = Path(backup_dir)
    
    ensure_directory(backup_dir)
    
    # Generate backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    shutil.copy2(file_path, backup_path)
    
    return backup_path

def cleanup_old_files(directory: Union[str, Path], pattern: str = "*", max_age_days: int = 30) -> List[Path]:
    """
    Clean up old files
    
    Args:
        directory: Directory path
        pattern: File pattern
        max_age_days: Maximum retention days
        
    Returns:
        List of deleted files
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    deleted_files = []
    current_time = datetime.now()
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.days > max_age_days:
                file_path.unlink()
                deleted_files.append(file_path)
    
    return deleted_files

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size (in bytes)
    
    Args:
        file_path: File path
        
    Returns:
        File size
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size

def format_file_size(size_bytes: int) -> str:
    """
    Format file size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

class FileManager:
    """File manager class"""
    
    def __init__(self, base_directory: Union[str, Path]):
        """
        Initialize file manager
        
        Args:
            base_directory: Base directory
        """
        self.base_directory = Path(base_directory)
        ensure_directory(self.base_directory)
    
    def get_path(self, *path_parts: str) -> Path:
        """
        Get path relative to base directory
        
        Args:
            *path_parts: Path components
            
        Returns:
            Full path
        """
        return self.base_directory.joinpath(*path_parts)
    
    def save_data(self, data: Any, filename: str, format: str = "json") -> Path:
        """
        Save data
        
        Args:
            data: Data to save
            filename: File name
            format: File format (json or csv)
            
        Returns:
            Saved file path
        """
        if format == "json":
            file_path = self.get_path(f"{filename}.json")
            save_json(data, file_path)
        elif format == "csv":
            file_path = self.get_path(f"{filename}.csv")
            save_csv(data, file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path
    
    def load_data(self, filename: str, format: str = "json") -> Any:
        """
        Load data
        
        Args:
            filename: File name (without extension)
            format: File format (json or csv)
            
        Returns:
            Loaded data
        """
        if format == "json":
            file_path = self.get_path(f"{filename}.json")
            return load_json(file_path)
        elif format == "csv":
            file_path = self.get_path(f"{filename}.csv")
            return load_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def list_files(self, pattern: str = "*") -> List[Path]:
        """
        List files
        
        Args:
            pattern: File pattern
            
        Returns:
            List of files
        """
        return list(self.base_directory.glob(pattern))
    
    def cleanup(self, max_age_days: int = 30) -> List[Path]:
        """
        Clean up old files
        
        Args:
            max_age_days: Maximum retention days
            
        Returns:
            List of deleted files
        """
        return cleanup_old_files(self.base_directory, max_age_days=max_age_days)

if __name__ == "__main__":
    # Test file operation functionality
    import tempfile
    
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Test directory: {temp_dir}")
        
        # Test file manager
        fm = FileManager(temp_dir)
        
        # Test save and load JSON
        test_data = {"name": "test", "value": 123, "items": [1, 2, 3]}
        json_path = fm.save_data(test_data, "test_data", "json")
        print(f"Saved JSON: {json_path}")
        
        loaded_data = fm.load_data("test_data", "json")
        print(f"Loaded JSON: {loaded_data}")
        
        # Test save and load CSV
        csv_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]
        csv_path = fm.save_data(csv_data, "test_data", "csv")
        print(f"Saved CSV: {csv_path}")
        
        loaded_csv = fm.load_data("test_data", "csv")
        print(f"Loaded CSV: {loaded_csv}")
        
        # Test list files
        files = fm.list_files()
        print(f"File list: {files}")
        
        print("File operation test completed")
