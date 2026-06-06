"""
server.py — Dedicated MCP Server with HTTP/SSE Transport.

A standalone FastMCP server providing shared context store, message bus,
and embedding store for inter-agent communication. Runs inside the same
Docker container as Streamlit, managed by supervisord.

Evolved from the reference project's mcp_server.py, adapted to use
HTTP/SSE transport (streamable-http) instead of stdio.
"""

import json
import uuid
import math
import time
import logging
from typing import Any

import jsonschema
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MCP] %(message)s")

# ── MCP Server Initialization ──────────────────────────────────────────────
mcp = FastMCP(
    name="knowledge-assistant-mcp",
    instructions=(
        "Shared-state MCP server for the Multi-Agent Knowledge Assistant. "
        "Provides: context store, message bus, and a vector embedding index "
        "for inter-agent communication via the Model Context Protocol."
    ),
    host="0.0.0.0",
    port=8100,
)

# ── JSON Schemas (validated on every write) ───────────────────────────────
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

# ── In-Memory Shared Stores ────────────────────────────────────────────────
_context_store:   dict[str, dict] = {}   # key → {key, value, agent_id}
_message_bus:     list[dict]      = []   # ordered broadcast log
_embedding_store: dict[str, dict] = {}   # doc_id → {embedding, text, metadata}


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL GROUP 1 – Context Store
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def store_context(key: str, value: str, agent_id: str) -> str:
    """
    Write a key-value pair to the shared context store.
    Any agent can later retrieve this value by key.
    Value is JSON-serialised by convention.
    """
    entry = {"key": key, "value": value, "agent_id": agent_id}
    try:
        jsonschema.validate(entry, CONTEXT_ENTRY_SCHEMA)
        _context_store[key] = entry
        logger.info(f"Context stored: key='{key}' by agent='{agent_id}'")
        return json.dumps({"status": "ok", "key": key, "written_by": agent_id})
    except jsonschema.ValidationError as exc:
        logger.error(f"Context validation error: {exc.message}")
        return json.dumps({"error": exc.message})


@mcp.tool()
def get_context(key: str) -> str:
    """Retrieve a value from the shared context store by key."""
    if key not in _context_store:
        return json.dumps({"error": f"Key '{key}' not found"})
    return json.dumps(_context_store[key])


@mcp.tool()
def list_context_keys() -> str:
    """List all keys currently held in the shared context store."""
    return json.dumps({"keys": list(_context_store.keys()),
                       "count": len(_context_store)})


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL GROUP 2 – Message Bus (Agent-to-Agent Communication)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def broadcast_message(
    agent_id:     str,
    message_type: str,
    content:      str,          # JSON-encoded dict
    correlation_id: str = ""
) -> str:
    """
    Post a message to the shared message bus.
    All other agents can poll this bus via read_messages().
    Message is validated against AGENT_MESSAGE_SCHEMA before acceptance.
    """
    msg: dict[str, Any] = {
        "agent_id":       agent_id,
        "type":           message_type,
        "content":        json.loads(content),
        "timestamp":      time.time(),
        "correlation_id": correlation_id or str(uuid.uuid4())
    }
    try:
        jsonschema.validate(msg, AGENT_MESSAGE_SCHEMA)
        _message_bus.append(msg)
        logger.info(f"Message broadcast: agent='{agent_id}', type='{message_type}'")
        return json.dumps({
            "status":         "sent",
            "correlation_id": msg["correlation_id"],
            "bus_size":       len(_message_bus)
        })
    except jsonschema.ValidationError as exc:
        logger.error(f"Message validation error: {exc.message}")
        return json.dumps({"error": exc.message})


@mcp.tool()
def read_messages(caller_agent_id: str, limit: int = 10) -> str:
    """
    Read up to `limit` recent messages from the bus, excluding own messages.
    Agents use this to discover work completed by peers.
    """
    peer_msgs = [
        m for m in _message_bus[-50:]
        if m["agent_id"] != caller_agent_id
    ]
    return json.dumps(peer_msgs[-limit:])


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL GROUP 3 – Embedding Store (Semantic Search)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def upsert_embedding(
    doc_id:    str,
    text:      str,
    embedding: str,           # JSON-encoded list[float]
    metadata:  str = "{}"     # JSON-encoded dict
) -> str:
    """
    Add or update a document + vector in the shared knowledge base.
    Research Agent populates this; Analysis / Writer Agents query it.
    """
    _embedding_store[doc_id] = {
        "doc_id":    doc_id,
        "text":      text,
        "embedding": json.loads(embedding),
        "metadata":  json.loads(metadata)
    }
    logger.info(f"Embedding upserted: doc_id='{doc_id}', store_size={len(_embedding_store)}")
    return json.dumps({"status": "upserted", "doc_id": doc_id,
                       "store_size": len(_embedding_store)})


@mcp.tool()
def semantic_search(
    query_embedding: str,    # JSON-encoded list[float]
    top_k: int = 3,
    threshold: float = 0.0
) -> str:
    """
    Return top-k documents by cosine similarity.
    Shared index means any agent can query without redundant embedding work.
    """
    def _cos_sim(a: list, b: list) -> float:
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    q_vec = json.loads(query_embedding)
    hits  = [
        {"doc_id":   d["doc_id"],
         "score":    round(_cos_sim(q_vec, d["embedding"]), 4),
         "text":     d["text"][:300],
         "metadata": d["metadata"]}
        for d in _embedding_store.values()
    ]
    hits = [h for h in hits if h["score"] >= threshold]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return json.dumps(hits[:top_k])


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Retrieve the full text of a stored document."""
    if doc_id not in _embedding_store:
        return json.dumps({"error": f"Document '{doc_id}' not found"})
    d = _embedding_store[doc_id]
    return json.dumps({"doc_id": d["doc_id"], "text": d["text"],
                        "metadata": d["metadata"]})


# ── Run as HTTP/SSE MCP transport ──────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting MCP server on port 8100 (streamable-http transport)")
    mcp.run(transport="streamable-http")
