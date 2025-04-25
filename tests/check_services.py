#!/usr/bin/env python
"""
Script to check if required services (Qdrant and Ollama) are running.
Use this before running the tests to verify the environment is properly set up.
"""

import requests
import sys
import time
import argparse


def check_qdrant(url: str, timeout: int = 5) -> bool:
    """Check if Qdrant is running at the specified URL."""
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        if response.status_code == 200:
            print(f"✅ Qdrant is running at {url}")
            return True
        else:
            print(f"❌ Qdrant returned unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to Qdrant at {url}")
        return False
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
        return False


def check_ollama(url: str, timeout: int = 5) -> bool:
    """Check if Ollama is running at the specified URL."""
    try:
        response = requests.get(f"{url.rstrip('/')}/api/version", timeout=timeout)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"✅ Ollama is running at {url} (version: {version})")
            return True
        else:
            print(f"❌ Ollama returned unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to Ollama at {url}")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def check_ollama_model(url: str, model: str, timeout: int = 5) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
            if model in available_models:
                print(f"✅ Model '{model}' is available in Ollama")
                return True
            else:
                print(f"❌ Model '{model}' not found in Ollama")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   To install: ollama pull {model}")
                return False
        else:
            print(f"❌ Failed to get models list: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking model availability: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check if required services are running")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", default="mxbai-embed-large:latest", help="Required Ollama model")
    parser.add_argument("--timeout", type=int, default=5, help="Connection timeout in seconds")
    args = parser.parse_args()

    print("Checking required services...")
    qdrant_ok = check_qdrant(args.qdrant_url, args.timeout)
    ollama_ok = check_ollama(args.ollama_url, args.timeout)
    
    model_ok = False
    if ollama_ok:
        model_ok = check_ollama_model(args.ollama_url, args.model, args.timeout)
    
    print("\nSummary:")
    print(f"Qdrant: {'✅ Ready' if qdrant_ok else '❌ Not available'}")
    print(f"Ollama: {'✅ Ready' if ollama_ok else '❌ Not available'}")
    print(f"Model '{args.model}': {'✅ Ready' if model_ok else '❌ Not available'}")
    
    if qdrant_ok and ollama_ok and model_ok:
        print("\n✅ All services are running and ready for tests!")
        return 0
    else:
        print("\n❌ Some services are not available. Tests may fail.")
        return 1


if __name__ == "__main__":
    sys.exit(main())