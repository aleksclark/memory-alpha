from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from memory_alpha.settings import DEFAULT_CONTEXT_LEVELS


def check_levels(context_levels: List[str]):
    # Validate all entries in context_levels against the allowed values
    invalid_values = (
        [level for level in context_levels if level not in DEFAULT_CONTEXT_LEVELS]
        if context_levels
        else []
    )
    if invalid_values:
        raise ValueError(
            f"Invalid context level(s): {invalid_values}. "
            f"Allowed values are: {DEFAULT_CONTEXT_LEVELS}"
        )
    return True


class QueryMemoryParams(BaseModel):
    prompt: str = Field(..., description="The full prompt to answer")
    max_tokens: Optional[int] = Field(
        1000, ge=50, le=4096, description="Cap on total tokens in returned context"
    )
    # Renamed attribute for clarity and used introduced constant for the default
    context_levels: Optional[List[str]] = Field(
        default_factory=lambda: DEFAULT_CONTEXT_LEVELS,
        description="Context levels to retrieve memories for",
    )

    filter: Optional[Dict[str, object]] = Field(
        None, description="Optional payload filter (e.g. by repo_path)"
    )
    k: Optional[int] = Field(
        24, ge=1, le=100, description="How many hits per level before token-capping"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_context_levels(cls, data: Any) -> Any:
        if isinstance(data, dict) and "context_levels" in data:
            check_levels(data.get("context_levels"))
        return data


class Chunk(BaseModel):
    level: str
    repo_path: str
    context: str
    score: Optional[float] = Field(
        default=None, description="Optional relevance score for bookkeeping"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_level(cls, data: Any) -> Any:
        if isinstance(data, dict) and "level" in data:
            level = data.get("level")
            if isinstance(level, str) and level not in DEFAULT_CONTEXT_LEVELS:
                raise ValueError(
                    f"Invalid context level: {level}. "
                    f"Allowed values are: {DEFAULT_CONTEXT_LEVELS}"
                )
        return data


class StoreMemoryParams(BaseModel):
    commit_id: str = Field(
        ..., description="SHA or unique ID of the commit being indexed"
    )
    diff: Optional[str] = Field(
        None, description="Optional raw git diff string (unified format)"
    )
    chunks: Optional[List[Chunk]] = Field(
        None, description="List of raw context chunks with metadata"
    )
    repo_root: Optional[str] = Field(
        ".", description="Repo root, used for diff parsing if needed"
    )
