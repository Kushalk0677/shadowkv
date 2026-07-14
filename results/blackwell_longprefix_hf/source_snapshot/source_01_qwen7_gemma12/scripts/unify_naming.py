#!/usr/bin/env python3
"""
Script to unify naming conventions across the codebase.
Replaces deprecated terms with standardized ones.
"""

import os
import re
import sys
from pathlib import Path

# Mapping of old terms to new standardized terms
NAMING_MAP = {
    'prefill_ms_per_token': 'prefill_ms_per_token',
    'past_key_values': 'past_key_values',
    'reuse_overhead_ms': 'reuse_overhead_ms',
    'prefill_cost_ms': 'prefill_cost_ms',
    'memory_bytes': 'memory_bytes',  # Keep as is for now
}

# Files/directories to exclude
EXCLUDE_PATTERNS = {
    '.git',
    '__pycache__',
    '.pytest_cache',
    'node_modules',
    'venv',
    '.venv',
    'env',
    '.env',
    'dist',
    'build',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '*.egg-info',
}

def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from processing."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in str(path) or path.match(pattern):
            return True
    return False

def replace_in_file(file_path: Path, replacements: dict) -> int:
    """Replace terms in a single file and return the number of replacements made."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        for old, new in replacements.items():
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(old)}\b'
            content = re.sub(pattern, new, content)
        
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return 1
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def unify_naming(root_dir: Path = Path('.')) -> None:
    """Unify naming conventions across the codebase."""
    print("Starting naming convention unification...")
    print(f"Root directory: {root_dir}")
    print(f"Replacements: {NAMING_MAP}")
    
    total_files = 0
    total_replacements = 0
    
    # Process Python files
    for py_file in root_dir.rglob('*.py'):
        if should_exclude(py_file):
            continue
            
        total_files += 1
        replacements = replace_in_file(py_file, NAMING_MAP)
        total_replacements += replacements
        
        if replacements > 0:
            print(f"  Updated {py_file}: {replacements} replacements")
    
    # Process other text files (config, markdown, etc.)
    text_extensions = {'.yaml', '.yml', '.md', '.txt', '.json', '.toml', '.cfg', '.ini'}
    for text_file in root_dir.rglob('*'):
        if text_file.is_file() and text_file.suffix in text_extensions and not should_exclude(text_file):
            total_files += 1
            replacements = replace_in_file(text_file, NAMING_MAP)
            total_replacements += replacements
            
            if replacements > 0:
                print(f"  Updated {text_file}: {replacements} replacements")
    
    print(f"\nCompleted naming unification:")
    print(f"  Files processed: {total_files}")
    print(f"  Total replacements: {total_replacements}")
    
    if total_replacements == 0:
        print("  No replacements were made.")

if __name__ == '__main__':
    # Get root directory from command line or use current directory
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    
    # Verify the directory exists
    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)
    
    if not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        sys.exit(1)
    
    unify_naming(root_dir)