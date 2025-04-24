0 · Purpose & Scope
The server gives LLM agents two core abilities:

query_context – retrieve a bounded “evidence‑pack” of code chunks (sig, section, file) that best answer a prompt.

index_update – push incremental code changes so the underlying Qdrant index stays fresh.

Both functions are exposed through JSON‑RPC 2.0 over HTTP POST (/messages) and HTTP + SSE (/sse) to satisfy FastAgent’s defaults.
The server combines a chunker, embedder, router, & cache behind those two calls.


4 · Internal Components
4.1 Chunker
Parses Python, TS/JS, Go, and Markdown:
* L1 sig = first line + docstring;
* L2 section = heuristic split on blank line ≤ 160 tokens;
* L3 file = rest of file ≤ 1200 tokens.

Emits (repo_path, level, code, loc_start, loc_end) tuples.

4.2 Embedder
text-embedding‑3‑small (1536 dims).

Batches up to 96 chunks / call.

Retries (exponential) up to 3 × before failing index_update.

4.3 Qdrant Client
One collection (dev_context).

HNSW, distance="Cosine", on_disk=true.

Payload fields: repo_path, level, commit_id, loc, timestamp.

4.4 Router
Vector search per level → collect top‑k.

Deduplicate identical code string.

Stop adding chunks when token count ≥ max_tokens; if overflow, run 1‑pass GPT summary to reach cap, set truncated=true.

4.5 Cache
In‑process LRU (maxsize=256, TTL 10 min).

Key = SHA256(prompt + levels + filter + k + max_tokens).

4.6 Back‑Pressure & SLA
Qdrant call timeout: 3 s (retry 3 ×).

If total time ≥ 4 min, keep streaming (heartbeat) until finish or call hits 300 s FastAPI timeout.
