#!/usr/bin/env python
"""
Debug script to print all loaded settings and their sources.
Run this script to see all the settings that will be applied to the application.
"""

import sys
import os
from typing import Any, Dict, List, Optional, Tuple
from pprint import pprint

from memory_alpha.settings import settings, DEFAULT_CONTEXT_LEVELS


def get_setting_source(setting_name: str) -> Tuple[str, Any]:
    """Determine the source of a setting (env var or default)."""
    env_var_name = setting_name.upper()
    if env_var_name in os.environ:
        return "environment", os.environ[env_var_name]
    
    if os.path.exists(".env"):
        with open(".env", "r") as env_file:
            for line in env_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        name, value = line.split("=", 1)
                        if name.strip() == env_var_name:
                            return ".env file", value.strip()
    
    return "default", getattr(settings, setting_name)


def format_setting(name: str, value: Any, source: str) -> str:
    """Format a setting for display."""
    value_str = str(value)
    if isinstance(value, str) and len(value) > 20 and "api_key" in name.lower():
        # Mask API keys for security
        value_str = value_str[:5] + "..." + value_str[-5:] if len(value_str) > 10 else "***"
    
    return f"{name:<25} = {value_str:<40} (from {source})"


def main() -> None:
    """Display all settings and their sources."""
    print("\n==== Memory Alpha Settings ====\n")
    
    # Get all settings attributes (excluding methods and special attributes)
    setting_names = [attr for attr in dir(settings) 
                     if not attr.startswith('_') and not callable(getattr(settings, attr))]
    
    # Group settings by category
    categories = {
        "Server Settings": ["qdrant_url", "openai_api_key"],
        "Model Settings": ["embed_model", "embed_dim"],
        "Collection Names": ["cluster_collection", "chunk_collection"],
        "Default Parameters": ["default_max_tokens", "default_k", "default_context_levels"],
    }
    
    # Print settings by category
    for category, names in categories.items():
        print(f"\n## {category}")
        for name in names:
            if name in setting_names:
                source_type, source_value = get_setting_source(name)
                setting_value = getattr(settings, name)
                print(format_setting(name, setting_value, source_type))
                setting_names.remove(name)
    
    # Print any remaining settings not in a category
    if setting_names:
        print("\n## Other Settings")
        for name in setting_names:
            # Skip internal pydantic settings
            if not name.startswith("model_") and not name.endswith("_fields"):
                source_type, source_value = get_setting_source(name)
                setting_value = getattr(settings, name)
                print(format_setting(name, setting_value, source_type))
    
    # Print special constants
    print("\n## Constants")
    print(f"DEFAULT_CONTEXT_LEVELS      = {DEFAULT_CONTEXT_LEVELS}")
    
    print("\n==== End of Settings ====\n")


if __name__ == "__main__":
    main()