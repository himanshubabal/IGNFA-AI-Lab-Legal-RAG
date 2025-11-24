#!/usr/bin/env python3
"""Test script to verify prompt imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from raganything.prompt import load_prompt_from_file, get_system_prompt, get_custom_prompt
    print("✅ All imports successful!")
    print(f"  - load_prompt_from_file: {load_prompt_from_file}")
    print(f"  - get_system_prompt: {get_system_prompt}")
    print(f"  - get_custom_prompt: {get_custom_prompt}")
    
    # Test loading prompt
    prompt = load_prompt_from_file("prompt.md")
    if prompt:
        print(f"\n✅ Custom prompt loaded ({len(prompt)} characters)")
    else:
        print("\nℹ️  No custom prompt.md found, using default")
        
    # Test system prompt
    default_prompt = get_system_prompt()
    print(f"\n✅ Default system prompt available ({len(default_prompt)} characters)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the virtual environment: source venv/bin/activate")
    print("2. Reinstall package: pip install -e .")
    print("3. Clear cache: find . -type d -name __pycache__ -exec rm -r {} +")
    sys.exit(1)

