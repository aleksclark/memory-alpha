#!/usr/bin/env python
"""Test script for settings changes"""

import sys
import os

# Get the full path to the settings.py file
settings_path = os.path.join(os.getcwd(), "src", "memory_alpha", "settings.py")
print(f"Settings path: {settings_path}")

# Copy the content of the settings file
with open(settings_path, "r") as f:
    content = f.read()
    print("\nSettings content:")
    print(content)

# Check if our properties are defined
if "@property" in content and "collection_prefix" in content:
    print("\nCollection prefix and property decorators found in the file")
else:
    print("\nWARNING: Property decorators or collection_prefix not found!")