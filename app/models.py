"""
models.py — Pydantic v2 schemas for the Multi-Agent Knowledge Assistant.

Shared data models used across RAG pipeline, CrewAI agents, MCP context,
LangGraph supervisor, and Streamlit UI.
"""

import uuid
import time
import operator
from typing import Optional, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
#  RAG Pipeline Models
# ═══════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Incoming user query with optional intent classification."""
    query: str = Field(..., min_length=1, description="User's question or request")
    intent: Optional[str] = Field(
        default=None,
        description="Classified intent: research | summarize | qa | cite"
    )
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session thread ID for conversation memory"
    )


class RetrievedChunk(BaseModel):
    """Single chunk retrieved from the vector store."""
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source document filename or URL")
    chunk_index: int = Field(..., ge=0, description="Position of chunk in source document")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    metadata: dict = Field(default_factory=dict, description="Additional chunk metadata")


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Context Models
# ═══════════════════════════════════════════════════════════════════════════

class MCPContext(BaseModel):
    """MCP message envelope for inter-agent context sharing."""
    context_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique context identifier"
    )
    agent_id: str = Field(..., description="Agent that created this context")
    source_doc: str = Field(default="", description="Source document reference")
    chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved chunks included in context"
    )
    embeddings: list[list[float]] = Field(
        default_factory=list,
        description="Embedding vectors for the chunks"
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp of context creation"
    )


class CitationModel(BaseModel):
    """Citation reference attached to an answer claim."""
    filename: str = Field(..., description="Source document filename")
    chunk_index: int = Field(..., ge=0, description="Chunk position in source")
    text_snippet: str = Field(
        default="",
        description="Short excerpt from the cited chunk"
    )

    def format(self) -> str:
        """Format citation as [filename · chunk_index]."""
        return f"[{self.filename} · {self.chunk_index}]"


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Message Schemas (for jsonschema validation)
# ═══════════════════════════════════════════════════════════════════════════

AGENT_MESSAGE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["agent_id", "type", "content", "timestamp"],
    "additionalProperties": False,
    "properties": {
        "agent_id":       {"type": "string", "minLength": 1},
        "type":           {"type": "string",
                           "enum": ["query", "result", "context_update", "broadcast"]},
        "content":        {"type": "object"},
        "timestamp":      {"type": "number"},
        "correlation_id": {"type": "string"}
    }
}

CONTEXT_ENTRY_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["key", "value", "agent_id"],
    "properties": {
        "key":      {"type": "string", "minLength": 1},
        "value":    {},                                      # any type allowed
        "agent_id": {"type": "string"},
        "ttl":      {"type": "number", "description": "seconds until expiry"}
    }
}


# ═══════════════════════════════════════════════════════════════════════════
#  LangGraph State Schema
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """LangGraph shared state flowing through the supervisor StateGraph."""
    query: str
    intent: str                                          # research | summarize | qa | cite
    retrieved_chunks: list[dict]
    mcp_context: dict
    crew_result: str
    final_answer: str
    chat_history: Annotated[list[dict], operator.add]     # accumulates across turns


# ═══════════════════════════════════════════════════════════════════════════
#  Custom Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class MCPValidationError(Exception):
    """Raised when an MCP message fails jsonschema validation."""
    pass
