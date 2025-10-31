"""
Path utilities for finding project root and resolving relative paths.

Allows scripts to find config/, data/, results/ directories regardless of where
they're run from.
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory (chexagent_chexpert_eval/).
    
    Assumes this file is at: src/utils/path_utils.py
    Project root is 2 levels up.
    """
    return Path(__file__).parent.parent.parent


def get_config_path(relative_path: str) -> Path:
    """Get absolute path to a config file."""
    return get_project_root() / "config" / relative_path


def get_data_path(relative_path: str) -> Path:
    """Get absolute path to a data file."""
    return get_project_root() / "data" / relative_path


def get_results_path(relative_path: str) -> Path:
    """Get absolute path to a results file."""
    return get_project_root() / "results" / relative_path


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path that might be relative to project root.
    
    If path starts with 'config/', 'data/', or 'results/', resolves relative to project root.
    Otherwise, treats as absolute or relative to current working directory.
    """
    path = Path(path_str)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # If starts with config/, data/, results/, resolve relative to project root
    parts = path.parts
    if len(parts) > 0:
        if parts[0] == "config":
            return get_config_path(str(Path(*parts[1:])))
        elif parts[0] == "data":
            return get_data_path(str(Path(*parts[1:])))
        elif parts[0] == "results":
            return get_results_path(str(Path(*parts[1:])))
    
    # Otherwise, return as-is (relative to CWD)
    return path

