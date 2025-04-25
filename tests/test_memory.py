"""
Tests for Memory Alpha query_memory and store_memory functions.

These tests require:
1. A running Qdrant server (default: http://localhost:6333)
2. A running Ollama server with mxbai-embed-large (default: http://localhost:11434)
"""

import pytest
import asyncio
from typing import Dict, List, Any

from memory_alpha.server import store_memory, query_memory
from memory_alpha.settings import settings, DEFAULT_CONTEXT_LEVELS


@pytest.mark.asyncio
async def test_store_memory_basic(sample_store_params):
    """Test that store_memory can store chunks without errors."""
    result = await store_memory(sample_store_params)
    
    # Check that the correct number of chunks were indexed
    assert result["indexed"] == len(sample_store_params["chunks"])
    assert result["removed"] == 0
    
    # duration_ms might be present but we don't test its value


@pytest.mark.asyncio
async def test_query_with_no_data():
    """Test query_memory with empty database."""
    # This test intentionally queries before storing any data
    query_params = {
        "prompt": "How do I normalize a vector?",
        "max_tokens": 500,
    }
    
    result = await query_memory(query_params)
    
    # We should get a valid response even with no data
    assert "chunks" in result
    assert "tokens" in result
    assert "truncated" in result
    assert isinstance(result["chunks"], list)
    
    # Since no data is stored, we expect 0 chunks
    assert len(result["chunks"]) == 0
    assert result["tokens"] == 0
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_store_and_query_full_cycle(sample_store_params):
    """Test the full cycle of storing and then querying memory."""
    # Step 1: Store sample data
    store_result = await store_memory(sample_store_params)
    assert store_result["indexed"] > 0, "Failed to index any chunks"
    
    # Step 2: Query with a related prompt
    query_params = {
        "prompt": "How do I normalize a vector in Python?",
        "max_tokens": 1000,
    }
    
    query_result = await query_memory(query_params)
    
    # We should get results back
    assert len(query_result["chunks"]) > 0, "No chunks returned from query"
    
    # The most relevant chunk should be about vector normalization
    found_relevant_chunk = False
    for chunk in query_result["chunks"]:
        if "normalize_vector" in chunk["context"] and "np.linalg.norm" in chunk["context"]:
            found_relevant_chunk = True
            break
    
    assert found_relevant_chunk, "Failed to retrieve the relevant chunk about vector normalization"
    assert query_result["tokens"] > 0, "No tokens were counted in the result"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level,expected_snippet",
    [
        ("function_signature", "def add(a, b)"),
        ("file_section", "class Calculator"),
        ("file", "import numpy as np"),
    ],
)
async def test_querying_by_level(sample_store_params, level, expected_snippet):
    """Test that querying by specific levels works correctly."""
    # Step 1: Store sample data
    await store_memory(sample_store_params)
    
    # Step 2: Query for a specific level only
    query_params = {
        "prompt": f"Show me code related to {level}",
        "levels": [level],  # Only search in this level
        "max_tokens": 1000,
    }
    
    result = await query_memory(query_params)
    
    # Check that we got results
    assert len(result["chunks"]) > 0, f"No chunks returned for level '{level}'"
    
    # Check that all returned chunks are from the requested level
    for chunk in result["chunks"]:
        assert chunk["level"] == level, f"Received chunk with level '{chunk['level']}' when requesting '{level}'"


@pytest.mark.asyncio
async def test_token_limiting(sample_store_params):
    """Test that max_tokens parameter correctly limits the returned content."""
    # Step 1: Store sample data
    await store_memory(sample_store_params)
    
    # First query with a high token limit
    query_high = {
        "prompt": "Show me all code",
        "max_tokens": 10000,  # Very high limit
    }
    
    result_high = await query_memory(query_high)
    
    # Now query with a low token limit
    query_low = {
        "prompt": "Show me all code",
        "max_tokens": 50,  # Very low limit
    }
    
    result_low = await query_memory(query_low)
    
    # The low limit query should return fewer chunks or tokens
    assert result_low["tokens"] <= query_low["max_tokens"], "Token limit was not respected"
    assert len(result_low["chunks"]) <= len(result_high["chunks"]), "Token limiting didn't reduce chunk count"


@pytest.mark.asyncio
async def test_multiple_stores_affects_importance(sample_store_params):
    """Test that storing similar chunks multiple times affects their importance."""
    # Store the original chunks
    await store_memory(sample_store_params)
    
    # Store a duplicate of one of the chunks to increase its importance
    duplicate_chunk = {
        "commit_id": "test_commit_duplicate",
        "chunks": [sample_store_params["chunks"][0]],  # Use the first chunk again
        "repo_root": "/sample",
    }
    
    await store_memory(duplicate_chunk)
    
    # Query for something related to the duplicated chunk
    query_params = {
        "prompt": "How to add two numbers in Python?",
    }
    
    result = await query_memory(query_params)
    
    # The first result should be the one we duplicated
    assert len(result["chunks"]) > 0, "No chunks returned from query"
    first_chunk = result["chunks"][0]
    assert "def add(a, b)" in first_chunk["context"], "The duplicated chunk was not prioritized"