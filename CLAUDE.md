# Python Project Best Practices with uv

## Environment Management
- Use `uv` for faster dependency management: `uv pip install <package>`
- Create virtual environments with: `uv venv`
- Activate: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)

## Dependency Management
- Use `pyproject.toml` for project configuration
- Pin dependencies with `uv pip freeze > requirements.txt`
- Use `uv pip compile` for lockfiles

## Code Quality
- Run linting with: `uv run ruff check .`
- Run type checking with: `uv run mypy .`
- Format code with: `uv run ruff format .`

## Testing
- Write tests using `uv run pytest`
- Run tests with: `uv run pytest`
- Check coverage with: `uv run pytest --cov=src`
