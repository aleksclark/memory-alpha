#!/usr/bin/env python
"""
Script to ensure Ollama is running and has the required model.
"""

import argparse
import logging
import sys

import requests

from memory_alpha.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_ollama_running(url: str = None) -> bool:
    """Check if Ollama is running at the specified URL."""
    url = url or settings.ollama_url
    try:
        response = requests.get(f"{url.rstrip('/')}/api/version")
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            logger.info(f"Ollama is running (version: {version})")
            return True
        else:
            logger.error(f"Ollama returned unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama at {url}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        return False


def check_model_available(model: str = None, url: str = None) -> bool:
    """Check if the specified model is available in Ollama."""
    model = model or settings.embed_model
    url = url or settings.ollama_url

    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags")
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
            if model in available_models:
                logger.info(f"Model '{model}' is available in Ollama")
                return True
            else:
                logger.warning(
                    f"Model '{model}' not found in Ollama. Available models: {', '.join(available_models)}"
                )
                return False
        else:
            logger.error(f"Failed to get models list: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False


def pull_model(model: str = None, url: str = None) -> bool:
    """Pull the specified model from Ollama."""
    model = model or settings.embed_model
    url = url or settings.ollama_url

    try:
        logger.info(f"Pulling model '{model}' (this may take a while)...")
        response = requests.post(f"{url.rstrip('/')}/api/pull", json={"name": model})
        if response.status_code == 200:
            logger.info(f"Successfully pulled model '{model}'")
            return True
        else:
            logger.error(f"Failed to pull model: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return False


def ensure_ollama_ready(
    model: str = None, url: str = None, auto_pull: bool = True
) -> bool:
    """Ensure Ollama is running and has the required model."""
    model = model or settings.embed_model
    url = url or settings.ollama_url

    if not check_ollama_running(url):
        logger.error("Ollama is not running. Please start Ollama first.")
        return False

    if not check_model_available(model, url):
        if auto_pull:
            logger.info(f"Trying to pull model '{model}'...")
            if pull_model(model, url):
                return True
            else:
                logger.error(f"Failed to pull model '{model}'")
                return False
        else:
            logger.warning(
                f"Model '{model}' not available. You can pull it with: ollama pull {model}"
            )
            return False

    return True


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(
        description="Ensure Ollama is running with the required model"
    )
    parser.add_argument(
        "--model", help="The model to check/pull (default: from settings)"
    )
    parser.add_argument("--url", help="The Ollama API URL (default: from settings)")
    parser.add_argument(
        "--no-pull", action="store_true", help="Don't pull the model if it's missing"
    )
    args = parser.parse_args()

    success = ensure_ollama_ready(
        model=args.model, url=args.url, auto_pull=not args.no_pull
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
