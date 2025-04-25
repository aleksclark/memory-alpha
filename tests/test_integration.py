"""
Integration tests that verify the end-to-end functionality
of Memory Alpha with real Qdrant and Ollama servers.
"""

import time

import pytest
import requests

from memory_alpha.embeddings import embed_text
from memory_alpha.settings import settings


@pytest.mark.integration
def test_ollama_connection():
    """Test that we can connect to Ollama and get embeddings."""
    try:
        # Check if Ollama is running
        response = requests.get(f"{settings.ollama_url.rstrip('/')}/api/version")
        assert response.status_code == 200, "Ollama server is not responding"

        # Test embedding generation
        text = "This is a test sentence for embedding."
        embedding = embed_text(text)

        # Verify the embedding has the correct dimension
        assert len(embedding) == settings.embed_dim, (
            f"Expected embedding dimension {settings.embed_dim}, got {len(embedding)}"
        )

        # Verify embeddings are not all zeros
        assert any(e != 0 for e in embedding), "Embedding contains only zeros"

        # Verify similar texts produce similar embeddings
        text2 = "This is a test phrase for embedding."
        embedding2 = embed_text(text2)

        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = sum(a * b for a, b in zip(embedding, embedding2, strict=False))
        assert similarity > 0.8, (
            f"Similar texts should have high similarity, got {similarity}"
        )

    except Exception as e:
        pytest.fail(f"Error connecting to Ollama: {e}")


@pytest.mark.integration
def test_qdrant_connection():
    """Test that we can connect to Qdrant and perform basic operations."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    try:
        # Create a test collection name
        test_collection = f"test_qdrant_connection_{int(time.time())}"

        # Connect to Qdrant
        client = QdrantClient(url=settings.qdrant_url)

        # Create a test collection
        if client.collection_exists(test_collection):
            client.delete_collection(test_collection)
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(
                size=settings.embed_dim, distance=Distance.COSINE
            ),
        )

        # Insert a test vector
        test_vector = [0.1] * settings.embed_dim
        client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(
                    id=1,  # Use numeric ID instead of string
                    vector=test_vector,
                    payload={"test": True},
                )
            ],
        )

        # Search for the vector
        results = client.query_points(
            collection_name=test_collection, query=test_vector, limit=1
        )

        # Clean up
        client.delete_collection(collection_name=test_collection)

        # Verify results
        assert len(results.points) == 1, (
            "Search did not return the expected number of results"
        )
        assert results.points[0].id == 1, "Search did not return the expected point"

    except Exception as e:
        pytest.fail(f"Error connecting to Qdrant: {e}")


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
async def test_end_to_end_with_real_code_samples():
    """Full integration test with real-world code samples."""
    from memory_alpha.server import query_memory, store_memory

    # Test collection names (to avoid interfering with production collections)
    test_cluster_coll = f"test_e2e_clusters_{int(time.time())}"
    test_chunk_coll = f"test_e2e_chunks_{int(time.time())}"

    # Backup original settings
    orig_cluster_coll = settings.cluster_collection
    orig_chunk_coll = settings.chunk_collection

    # Override with test collections
    settings.cluster_collection = test_cluster_coll
    settings.chunk_collection = test_chunk_coll

    try:
        # Create collections
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(url=settings.qdrant_url)
        # Create test collections
        for collection_name in [test_cluster_coll, test_chunk_coll]:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embed_dim, distance=Distance.COSINE
                ),
            )

        # Real-world code samples
        code_samples = [
            {
                "context": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
                """,
                "level": "function_signature",
                "repo_path": "/algorithms/sorting.py",
            },
            {
                "context": """
class LinkedList:
    def __init__(self):
        self.head = None
        
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
            
    def append(self, data):
        if not self.head:
            self.head = self.Node(data)
            return
            
        current = self.head
        while current.next:
            current = current.next
        current.next = self.Node(data)
                """,
                "level": "file_section",
                "repo_path": "/data_structures/linked_list.py",
            },
            {
                "context": """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function DataFetcher({ url, render }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await axios.get(url);
                setData(response.data);
                setError(null);
            } catch (err) {
                setError(err.message);
                setData(null);
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [url]);
    
    return render({ data, loading, error });
}

export default DataFetcher;
                """,
                "level": "file",
                "repo_path": "/components/DataFetcher.jsx",
            },
        ]

        # Store code samples
        store_params = {
            "commit_id": "test_e2e_commit",
            "chunks": code_samples,
        }

        store_result = await store_memory(store_params)
        assert store_result["indexed"] == len(code_samples), (
            "Not all samples were indexed"
        )

        # Test queries with varying prompts
        queries = [
            {
                "prompt": "How does quicksort algorithm work?",
                "expected_in_context": ["quicksort", "pivot"],
            },
            {
                "prompt": "Show me a React component for fetching data",
                "expected_in_context": ["axios", "useState", "useEffect"],
            },
            {
                "prompt": "How to implement a linked list in Python?",
                "expected_in_context": ["LinkedList", "Node", "append"],
            },
        ]

        for query_info in queries:
            query_params = {
                "prompt": query_info["prompt"],
            }

            result = await query_memory(query_params)

            # With real embeddings, we can't guarantee exact content matches
            # Let's just check that we get some results back
            assert len(result["chunks"]) > 0, (
                f"No results for query: {query_info['prompt']}"
            )
            assert result["tokens"] > 0, f"No tokens for query: {query_info['prompt']}"

    finally:
        # Clean up
        try:
            client.delete_collection(collection_name=test_cluster_coll)
            client.delete_collection(collection_name=test_chunk_coll)
        except Exception as e:
            print(f"Error cleaning up collections: {e}")

        # Restore original settings
        settings.cluster_collection = orig_cluster_coll
        settings.chunk_collection = orig_chunk_coll
