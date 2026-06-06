"""
mcp_client.py — MCP HTTP Client.

Connects to the co-located MCP server (localhost:8100) for inter-agent
context sharing. Uses the MCP SDK's StreamableHTTPTransport for native
MCP protocol communication.
"""

import os
import json
import logging
import time
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import StreamableHTTPTransport

from app.models import RetrievedChunk, MCPContext, MCPValidationError

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8100")


class MCPClient:
    """
    Client for the co-located MCP server.

    Provides methods to store/retrieve context, broadcast messages,
    and validate MCP messages — all via the MCP protocol over HTTP.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or MCP_SERVER_URL
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[any] = None

    async def _ensure_session(self) -> ClientSession:
        """Lazily create and initialize an MCP session."""
        if self._session is None:
            mcp_url = f"{self.base_url}/mcp"
            logger.info(f"Connecting to MCP server at {mcp_url}")
            from mcp.client.streamable_http import streamable_http_client
            from contextlib import AsyncExitStack

            self._exit_stack = AsyncExitStack()
            
            # Enter streamable_http_client context
            read, write, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(mcp_url)
            )
            
            # Enter ClientSession context
            self._session = ClientSession(read, write)
            await self._exit_stack.enter_async_context(self._session)
            
            await self._session.initialize()
            logger.info("MCP session initialized successfully")
        return self._session

    async def _call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the text result."""
        session = await self._ensure_session()
        result = await session.call_tool(tool_name, arguments=arguments)
        return result.content[0].text


    # ── Context Store Operations ──────────────────────────────────────────

    async def store_context(self, key: str, value: str, agent_id: str) -> dict:
        """Write a key-value pair to the MCP shared context store."""
        result = await self._call_tool("store_context", {
            "key": key, "value": value, "agent_id": agent_id
        })
        return json.loads(result)

    async def get_context(self, key: str) -> dict:
        """Read a context value from the MCP server by key."""
        result = await self._call_tool("get_context", {"key": key})
        return json.loads(result)

    async def list_context_keys(self) -> dict:
        """List all keys in the MCP context store."""
        result = await self._call_tool("list_context_keys", {})
        return json.loads(result)

    # ── Message Bus Operations ────────────────────────────────────────────

    async def broadcast_message(
        self, agent_id: str, message_type: str, content: dict
    ) -> dict:
        """Post a message to the MCP shared message bus."""
        result = await self._call_tool("broadcast_message", {
            "agent_id": agent_id,
            "message_type": message_type,
            "content": json.dumps(content),
        })
        return json.loads(result)

    async def read_messages(self, agent_id: str, limit: int = 10) -> list:
        """Read peer messages from the MCP message bus."""
        result = await self._call_tool("read_messages", {
            "caller_agent_id": agent_id, "limit": limit
        })
        return json.loads(result)

    # ── High-Level Context Sharing ────────────────────────────────────────

    async def validate_and_share(
        self, chunks: list[RetrievedChunk], agent_id: str
    ) -> MCPContext:
        """
        Package retrieved chunks into an MCP context, validate, and share
        via the MCP server.

        Args:
            chunks: Retrieved chunks from the RAG pipeline.
            agent_id: The agent sharing the context.

        Returns:
            MCPContext with the stored context metadata.

        Raises:
            MCPValidationError: If the MCP server rejects the message.
        """
        # Build MCP context model
        mcp_context = MCPContext(
            agent_id=agent_id,
            source_doc=chunks[0].source if chunks else "unknown",
            chunks=chunks,
        )

        # Store in MCP context store
        store_result = await self.store_context(
            key=f"context:{mcp_context.context_id}",
            value=json.dumps({
                "context_id": mcp_context.context_id,
                "agent_id": agent_id,
                "source_doc": mcp_context.source_doc,
                "chunk_count": len(chunks),
                "created_at": mcp_context.created_at,
            }),
            agent_id=agent_id,
        )

        if "error" in store_result:
            raise MCPValidationError(
                f"MCP context storage failed: {store_result['error']}"
            )

        # Broadcast to message bus
        broadcast_result = await self.broadcast_message(
            agent_id=agent_id,
            message_type="context_update",
            content={
                "phase": "context_shared",
                "context_id": mcp_context.context_id,
                "chunk_count": len(chunks),
            },
        )

        if "error" in broadcast_result:
            raise MCPValidationError(
                f"MCP broadcast failed: {broadcast_result['error']}"
            )

        logger.info(
            f"MCP context shared: id={mcp_context.context_id}, "
            f"chunks={len(chunks)}, agent={agent_id}"
        )
        return mcp_context

    async def close(self):
        """Clean up the MCP session and transport."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

