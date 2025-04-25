import hashlib
import logging
import time
import uuid

from fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from memory_alpha.embeddings import embed_text
from memory_alpha.params import QueryMemoryParams, StoreMemoryParams
from memory_alpha.settings import settings

# Create server
mcp = FastMCP(
    "Memory Server",
    instructions=f"""
This tool is used to store and query memories about software projects.
Memories have a level, which can be used to filter them. The levels are #{settings.default_context_levels}.
When you analyze a project, you will collect memories at different levels. For example, if you look at a file called "Todos.ts"
you might store:
  1. A memory that this project is using prisma for database models and access
  2. A memory that this module provides functions for manipulating Todos
  3. A memory that this file depends on the prisma client library
  4. A memory that this file depends on another module for notifying the system of changes
  
As you make changes to the project, you will update memories with the new information.
Before making changes, you should query the server to see if there are any memories that might be relevant to your changes.

Higher level memories are more important than lower level memories. If you have a memory that is more important than another memory, assign it a higher importance value.
""",
)

# Initialize Qdrant client
qdrant = QdrantClient(url=settings.qdrant_url)


def ensure_collections_exist():
    """Ensure that the required Qdrant collections exist.

    If they don't exist, create them with the correct configuration.
    Returns True if all collections exist/were created successfully, False otherwise.
    """
    try:
        for collection_name in [settings.cluster_collection, settings.chunk_collection]:
            if not qdrant.collection_exists(collection_name):
                logger.info(
                    f"Creating collection '{collection_name}' as it doesn't exist"
                )
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embed_dim, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection '{collection_name}' created successfully")
            else:
                logger.debug(f"Collection '{collection_name}' already exists")
        return True
    except Exception as e:
        logger.error(f"Failed to create collections: {str(e)}")
        return False


def hash_chunk_id(path, level, context):
    """Generate a numeric ID for a chunk based on its content.

    Qdrant requires IDs to be either an unsigned integer or a UUID.
    We'll use a hash of the content and convert it to an integer.
    """
    # Create a hash of the content
    hash_obj = hashlib.sha256(f"{path}:{level}:{context}".encode())
    # Convert first 16 bytes (128 bits) of hash to integer
    # This should be enough to avoid collisions while fitting in uint64
    return int(hash_obj.hexdigest()[:16], 16)


@mcp.tool(description="Store chunks of context in the memory server")
async def store_memory(params: StoreMemoryParams):
    # Ensure collections exist before attempting to store data
    if not ensure_collections_exist():
        return {
            "error": "Failed to create necessary collections",
            "indexed": 0,
            "removed": 0,
            "duration_ms": 0,
        }

    commit_id = params.commit_id
    chunks = params.chunks or []
    indexed = 0

    to_upsert_chunks = []
    to_upsert_clusters = []

    try:
        for chunk in chunks:
            vec = embed_text(chunk.context)
            path = chunk.repo_path
            level = chunk.level
            cid = hash_chunk_id(path, level, chunk.context)

            # Search for cluster match
            hits = qdrant.query_points(
                collection_name=settings.cluster_collection,
                query=vec,
                limit=5,
                query_filter=Filter(
                    must=[FieldCondition(key="level", match=MatchValue(value=level))]
                ),
            )

            assigned_cluster = None
            # In new API, hits is a QueryResponse object with points attribute
            for point in hits.points:
                if point.score >= 0.85:
                    assigned_cluster = point
                    break

            # Generate a cluster ID - convert UUID to an integer
            cluster_id = int(uuid.uuid4().int % (2**63))
            if assigned_cluster:
                # Use existing cluster ID if available
                cluster_id = (
                    assigned_cluster.id
                    if isinstance(assigned_cluster.id, int)
                    else cluster_id
                )
                # Make sure the vector is not None
                if (
                    hasattr(assigned_cluster, "vector")
                    and assigned_cluster.vector is not None
                ):
                    vec = [
                        (v1 * assigned_cluster.payload["member_count"] + v2)
                        / (assigned_cluster.payload["member_count"] + 1)
                        for v1, v2 in zip(assigned_cluster.vector, vec, strict=False)
                    ]
                to_upsert_clusters.append(
                    PointStruct(
                        id=cluster_id,
                        vector=vec,
                        payload={
                            "level": level,
                            "member_count": assigned_cluster.payload["member_count"]
                            + 1,
                            "importance": assigned_cluster.payload.get(
                                "importance", 1.0
                            )
                            + 0.5,
                        },
                    )
                )
            else:
                to_upsert_clusters.append(
                    PointStruct(
                        id=cluster_id,
                        vector=vec,
                        payload={"level": level, "member_count": 1, "importance": 1.0},
                    )
                )

            to_upsert_chunks.append(
                PointStruct(
                    id=cid,
                    vector=vec,
                    payload={
                        "repo_path": path,
                        "level": level,
                        "context": chunk.context,
                        "cluster_id": cluster_id,  # This is now an integer
                        "commit_id": commit_id,
                        "access_count": 0,
                        "timestamp": time.time(),
                    },
                )
            )
            indexed += 1

        # Only perform upsert if we have points to add
        if to_upsert_clusters:
            qdrant.upsert(
                collection_name=settings.cluster_collection, points=to_upsert_clusters
            )
        if to_upsert_chunks:
            qdrant.upsert(
                collection_name=settings.chunk_collection, points=to_upsert_chunks
            )

    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        return {
            "error": f"Failed to store memory: {str(e)}",
            "indexed": 0,
            "removed": 0,
            "duration_ms": 0,
        }

    return {"indexed": indexed, "removed": 0, "duration_ms": 0}


@mcp.tool(description="Query the memory server for a given commit ID and chunks")
async def query_memory(params: QueryMemoryParams):
    # Ensure collections exist before attempting to query data
    if not ensure_collections_exist():
        return {
            "error": "Failed to create necessary collections",
            "chunks": [],
            "truncated": False,
            "tokens": 0,
        }

    prompt = params.prompt
    max_tokens = params.max_tokens or settings.default_max_tokens
    levels = params.context_levels or settings.default_context_levels
    k = params.k or settings.default_k

    try:
        vec = embed_text(prompt)

        # Search cluster centroids
        hits = qdrant.query_points(
            collection_name=settings.cluster_collection,
            query=vec,
            limit=10,
            query_filter=Filter(
                must=[FieldCondition(key="level", match=MatchValue(value=levels[0]))]
            ),
        )
    except Exception as e:
        logger.error(f"Error querying points: {str(e)}")
        return {
            "error": f"Failed to query memory: {str(e)}",
            "chunks": [],
            "truncated": False,
            "tokens": 0,
        }

    candidate_chunks = []
    try:
        # In new API, hits is a QueryResponse object with points attribute
        for cluster in hits.points:
            cluster_id = cluster.id  # Keep the original ID type (integer)
            cluster_chunks = qdrant.scroll(
                collection_name=settings.chunk_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="cluster_id", match=MatchValue(value=cluster_id)
                        ),
                        FieldCondition(key="level", match=MatchValue(value=levels[0])),
                    ]
                ),
                limit=k,
            )[0]
            candidate_chunks.extend(cluster_chunks)
    except Exception as e:
        logger.error(f"Error scrolling chunks: {str(e)}")
        return {
            "error": f"Failed to retrieve chunks: {str(e)}",
            "chunks": [],
            "truncated": False,
            "tokens": 0,
        }

    sorted_chunks = sorted(
        candidate_chunks,
        key=lambda x: (
            x.payload.get("access_count", 0) + x.payload.get("importance", 1.0)
        ),
        reverse=True,
    )

    pack = []
    token_count = 0
    for c in sorted_chunks:
        context = c.payload.get("context", "")
        context_tokens = len(context.split())  # rough approximation
        if token_count + context_tokens > max_tokens:
            break
        pack.append(
            {
                "repo_path": c.payload["repo_path"],
                "level": c.payload["level"],
                "context": context,
                "score": 1.0,  # placeholder
            }
        )
        token_count += context_tokens

    return {
        "chunks": pack,
        "truncated": token_count >= max_tokens,
        "tokens": token_count,
    }
