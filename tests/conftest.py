"""
Test fixtures for Memory Alpha tests.
"""

import uuid
from typing import Any, Dict, List

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from memory_alpha.ensure_ollama import ensure_ollama_ready
from memory_alpha.params import Chunk, StoreMemoryParams
from memory_alpha.settings import settings

# Don't define a custom event_loop fixture; use pytest-asyncio's built-in one
# Instead, we can configure it globally in pytest.ini


@pytest.fixture(scope="session")
def check_ollama():
    """Ensure Ollama is running and has the required model."""
    # This will raise an exception if Ollama is not available
    if not ensure_ollama_ready():
        pytest.skip("Ollama is not running or required model is not available")


@pytest.fixture(scope="session", autouse=True)
def setup_test_collections(check_ollama):
    """
    Set up test collections in Qdrant and clean them up after tests.

    This fixture creates unique collection names for tests to avoid
    interfering with existing collections.
    """
    # Generate unique collection names for this test run
    test_id = str(uuid.uuid4())[:8]
    test_cluster_collection = f"test_clusters_{test_id}"
    test_chunk_collection = f"test_chunks_{test_id}"

    # Override settings for tests
    original_cluster_collection = settings.cluster_collection
    original_chunk_collection = settings.chunk_collection
    settings.cluster_collection = test_cluster_collection
    settings.chunk_collection = test_chunk_collection

    # Create client
    qdrant = QdrantClient(url=settings.qdrant_url)

    # Create test collections
    for collection_name in [test_cluster_collection, test_chunk_collection]:
        # Delete collection if it exists
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)

        # Create new collection
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embed_dim, distance=Distance.COSINE
            ),
        )

    yield

    # Clean up
    try:
        qdrant.delete_collection(collection_name=test_cluster_collection)
        qdrant.delete_collection(collection_name=test_chunk_collection)
    except Exception as e:
        print(f"Error cleaning up test collections: {e}")

    # Restore original settings
    settings.cluster_collection = original_cluster_collection
    settings.chunk_collection = original_chunk_collection


@pytest.fixture
def sample_code_chunks() -> List[Dict[str, Any]]:
    """Sample code chunks for testing."""
    return [
        {
            "context": 'def add(a, b):\n    """Add two numbers and return the result."""\n    return a + b',
            "level": "function_signature",
            "repo_path": "/sample/math.py",
        },
        {
            "context": "class Calculator:\n    def __init__(self):\n        self.memory = 0\n        \n    def add(self, x):\n        self.memory += x\n        return self.memory",
            "level": "file_section",
            "repo_path": "/sample/calculator.py",
        },
        {
            "context": 'import numpy as np\n\ndef normalize_vector(v):\n    """Normalize a vector to unit length."""\n    norm = np.linalg.norm(v)\n    if norm == 0:\n        return v\n    return v / norm',
            "level": "file",
            "repo_path": "/sample/vector_utils.py",
        },
        {
            "context": "// React component for a todo list\nimport React, { useState } from 'react';\n\nconst TodoList = ({ initialTodos }) => {\n  const [todos, setTodos] = useState(initialTodos);\n  const [input, setInput] = useState('');\n\n  const addTodo = () => {\n    if (input) {\n      setTodos([...todos, { text: input, completed: false }]);\n      setInput('');\n    }\n  };\n\n  return (\n    <div>\n      <input\n        value={input}\n        onChange={(e) => setInput(e.target.value)}\n        placeholder=\"Add a todo\"\n      />\n      <button onClick={addTodo}>Add</button>\n      <ul>\n        {todos.map((todo, i) => (\n          <li key={i}>{todo.text}</li>\n        ))}\n      </ul>\n    </div>\n  );\n};\n\nexport default TodoList;",
            "level": "file",
            "repo_path": "/sample/TodoList.jsx",
        },
    ]


@pytest.fixture
def sample_store_params(sample_code_chunks):
    """Sample parameters for store_memory."""
    # Convert dictionary chunks to Chunk objects
    chunk_objects = [Chunk(**chunk) for chunk in sample_code_chunks]
    
    # Create a StoreMemoryParams object
    return StoreMemoryParams(
        commit_id="test_commit_123",
        chunks=chunk_objects,
        repo_root="/sample",
    )
