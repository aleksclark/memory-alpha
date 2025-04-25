# Memory Alpha - Model Context Protocol Server

Memory Alpha is a Model Context Protocol (MCP) server that gives LLM agents the ability to retrieve relevant code context and update the code index, using the FastMCP library.

## Features

- **query_context**: Retrieve bounded "evidence-packs" of code chunks that best answer a prompt
- **index_update**: Push incremental code changes to keep the underlying index fresh
- Built with FastMCP for minimal boilerplate and automatic SSE support
- Optimized for LLM agent interactions

## Installation

### Prerequisites

- Python 3.10 or higher
- [FastMCP](https://github.com/jlowin/fastmcp) (installed from GitHub)
- [Ollama](https://ollama.ai/) for embeddings
- Qdrant for vector storage

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/memory-alpha.git
cd memory-alpha

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install git+https://github.com/jlowin/fastmcp.git
uv pip install -e .

# Ensure Ollama is running and has the required model
./ensure_ollama.py
```

## Usage

### Starting the server

```bash
# HTTP/SSE mode
./run.py

# stdio mode (for use with modelcontextprotocol/inspector)
./run_stdio.py

# Or using fastmcp directly
fastmcp run --port 58765 --transport sse server.py  # HTTP/SSE mode
fastmcp run --transport stdio server.py             # stdio mode
```

The HTTP server will start on http://0.0.0.0:58765 by default.

### Validating the server

The project includes several validation scripts:

```bash
# Validate both HTTP and stdio modes
./validate_all.py

# Validate only HTTP mode
./simple_validation.py --start

# Validate only stdio mode
./validate_stdio.py
```

### API Tools

- **query_context**: Retrieve relevant code chunks for a prompt
- **index_update**: Update the code index with new file content
- **resource://health**: Health check endpoint
- **resource://docs/chunking**: Documentation about the chunking process

## Command Line Tools

When installed as a package, Memory Alpha provides several command-line tools:

- **memory-alpha**: Start the MCP server
- **memory-alpha-ensure-ollama**: Check if Ollama is running and has the required model
- **memory-alpha-debug-settings**: Display all current settings and their sources

These tools can be used with command-line arguments. For example:

```bash
# Start the server on a specific port
memory-alpha --port 8000

# Check for a different Ollama model
memory-alpha-ensure-ollama --model mxbai-embed-large-v2

# Show settings while overriding environment variables
OLLAMA_URL=http://other-server:11434 memory-alpha-debug-settings
```

Run any command with `--help` to see all available options.

### Example queries

```bash
# Using curl to query the health endpoint
curl http://localhost:9876/resources/resource://health

# Using the MCP Inspector to test the server
npx @modelcontextprotocol/inspector --cli http://localhost:9876 --method tools/list
```

## Project Structure

- `src/memory_alpha/server.py`: Main MCP server implementation
- `src/memory_alpha/settings.py`: Settings configuration using pydantic_settings
- `src/memory_alpha/params.py`: Input parameter schemas
- `.env`: Environment configuration (see below)

## Configuration

Memory Alpha uses environment variables for configuration, which can be set directly or through a `.env` file.

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your settings:
   ```
   # Server settings
   QDRANT_URL=http://localhost:6333
   OLLAMA_URL=http://localhost:11434
   
   # Model settings
   EMBED_MODEL=mxbai-embed-large
   
   # Collection names
   CLUSTER_COLLECTION=production_clusters
   CHUNK_COLLECTION=production_chunks
   ```

3. Install Ollama from [ollama.ai](https://ollama.ai/) and pull the embedding model:
   ```bash
   ollama pull mxbai-embed-large
   ```
   
   Alternatively, if you've installed the package, use the bundled command:
   ```bash
   memory-alpha-ensure-ollama
   ```
   
   Or run the script directly:
   ```bash
   ./ensure_ollama.py
   ```

All settings have sensible defaults and will only be overridden by environment variables if provided.

4. To view your current settings configuration, use one of the following:
   
   If you've installed the package:
   ```bash
   memory-alpha-debug-settings
   ```
   
   Or run the script directly:
   ```bash
   python debug_settings.py
   ```
   
   This will show all settings and their sources (default, environment variable, or .env file).

   You can also set environment variables directly when running the command:
   ```bash
   QDRANT_URL=http://another-server:6333 memory-alpha-debug-settings
   ```

## Development

### Running tests

```bash
pytest
```

### Linting and type checking

```bash
ruff check .
mypy .
```