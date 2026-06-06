"""
supervisor.py — LangGraph Supervisor StateGraph.

Orchestrates the multi-agent pipeline: classify intent → retrieve context
→ build MCP context → run CrewAI crew → format response with citations.
Uses MemorySaver for conversation memory across turns.
"""

import os
import json
import asyncio
import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.models import AgentState, RetrievedChunk
from app.rag import retrieve_context
from app.crew import build_crew
from app.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Shared Resources
# ═══════════════════════════════════════════════════════════════════════════

_llm = None
_mcp_client = None
_memory = MemorySaver()


def _get_llm() -> ChatOpenAI:
    """Lazily create the LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
        )
    return _llm


def _get_mcp_client() -> MCPClient:
    """Lazily create the MCP client."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


# ═══════════════════════════════════════════════════════════════════════════
#  Node 1 — Classify Intent
# ═══════════════════════════════════════════════════════════════════════════

def classify_intent(state: AgentState) -> dict:
    """
    Classify the user's query intent into one of four categories:
    research, summarize, qa, cite.
    """
    query = state["query"]
    llm = _get_llm()

    classification_prompt = f"""Classify the following user query into exactly one category.

Categories:
- research: The user wants to explore or investigate a topic broadly
- summarize: The user wants a summary of document content
- qa: The user is asking a specific factual question
- cite: The user wants citations or references for claims

Query: "{query}"

Respond with ONLY the category name (research, summarize, qa, or cite). Nothing else."""

    try:
        response = llm.invoke(classification_prompt)
        intent = response.content.strip().lower()

        # Validate the intent
        valid_intents = {"research", "summarize", "qa", "cite"}
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to 'qa'")
            intent = "qa"

        logger.info(f"Intent classified: '{intent}' for query: '{query[:50]}...'")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}", exc_info=True)
        intent = "qa"  # fallback

    return {
        "intent": intent,
        "chat_history": [{"role": "user", "content": query}],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Node 2 — Retrieve Context from RAG
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB for the user's query.
    """
    query = state["query"]
    logger.info(f"Retrieving context for: '{query[:50]}...'")

    try:
        chunks = retrieve_context(query, top_k=5)
        chunks_dicts = [c.model_dump() for c in chunks]
        logger.info(f"Retrieved {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        chunks_dicts = []

    return {"retrieved_chunks": chunks_dicts}


# ═══════════════════════════════════════════════════════════════════════════
#  Node 3 — Build MCP Context
# ═══════════════════════════════════════════════════════════════════════════

def build_mcp_node(state: AgentState) -> dict:
    """
    Package retrieved chunks into MCP context and share via MCP server.
    """
    chunks_dicts = state.get("retrieved_chunks", [])
    chunks = [RetrievedChunk(**c) for c in chunks_dicts]

    if not chunks:
        logger.warning("No chunks to share via MCP")
        return {"mcp_context": {"status": "no_chunks"}}

    mcp_client = _get_mcp_client()

    try:
        # Run async MCP operations in sync context
        loop = asyncio.new_event_loop()
        mcp_context = loop.run_until_complete(
            mcp_client.validate_and_share(chunks, agent_id="supervisor")
        )
        loop.close()

        context_dict = {
            "context_id": mcp_context.context_id,
            "agent_id": mcp_context.agent_id,
            "chunk_count": len(chunks),
            "created_at": mcp_context.created_at,
            "status": "shared",
        }
        logger.info(f"MCP context built: {mcp_context.context_id}")
    except Exception as e:
        logger.warning(f"MCP context sharing failed (non-fatal): {e}")
        context_dict = {"status": "fallback", "error": str(e)}

    return {"mcp_context": context_dict}


# ═══════════════════════════════════════════════════════════════════════════
#  Node 4 — Run CrewAI Agents
# ═══════════════════════════════════════════════════════════════════════════

def run_crew_node(state: AgentState) -> dict:
    """
    Execute the CrewAI specialist crew with retrieved context.
    """
    query = state["query"]
    intent = state.get("intent", "qa")
    chunks_dicts = state.get("retrieved_chunks", [])
    chunks = [RetrievedChunk(**c) for c in chunks_dicts]

    logger.info(f"Running crew: intent='{intent}', chunks={len(chunks)}")

    try:
        result = build_crew(query=query, context=chunks, intent=intent)
        logger.info(f"Crew completed, result length: {len(result)}")
    except Exception as e:
        logger.error(f"Crew execution failed: {e}", exc_info=True)
        # Graceful fallback: return partial result with error badge
        result = (
            f"⚠️ **Partial Result** — Agent processing encountered an error.\n\n"
            f"**Query:** {query}\n\n"
            f"**Retrieved Context ({len(chunks)} chunks):**\n"
        )
        for c in chunks[:3]:
            result += f"- [{c.source} · {c.chunk_index}]: {c.text[:200]}...\n"
        result += f"\n**Error:** {str(e)}"

    return {"crew_result": result}


# ═══════════════════════════════════════════════════════════════════════════
#  Node 5 — Format Response
# ═══════════════════════════════════════════════════════════════════════════

def format_response(state: AgentState) -> dict:
    """
    Format the final response with citations and update chat history.
    """
    crew_result = state.get("crew_result", "No response generated.")
    intent = state.get("intent", "qa")
    chunks_dicts = state.get("retrieved_chunks", [])

    # The crew should have already added citations, but ensure minimal formatting
    final_answer = crew_result

    # Add retrieval metadata footer if chunks were used
    if chunks_dicts:
        sources = set()
        for c in chunks_dicts:
            sources.add(c.get("source", "unknown"))
        source_list = ", ".join(sources)
        final_answer += f"\n\n---\n*Sources consulted: {source_list}*"

    logger.info(f"Response formatted: intent='{intent}', length={len(final_answer)}")

    return {
        "final_answer": final_answer,
        "chat_history": [{"role": "assistant", "content": final_answer}],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Build the StateGraph
# ═══════════════════════════════════════════════════════════════════════════

def _build_graph() -> StateGraph:
    """
    Construct the LangGraph supervisor StateGraph.

    Flow: START → classify_intent → retrieve → build_mcp → run_crew
                → format_response → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("build_mcp", build_mcp_node)
    graph.add_node("run_crew", run_crew_node)
    graph.add_node("format_response", format_response)

    # Wire edges — linear pipeline (intent influences crew behavior, not routing)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve")
    graph.add_edge("retrieve", "build_mcp")
    graph.add_edge("build_mcp", "run_crew")
    graph.add_edge("run_crew", "format_response")
    graph.add_edge("format_response", END)

    return graph


# Global database initialization flag
_db_initialized = False


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_query(query: str, thread_id: str) -> dict:
    """
    Execute the full supervisor pipeline for a user query.

    Args:
        query: The user's question or request.
        thread_id: Session thread ID for conversation memory.

    Returns:
        Dict with keys: final_answer, intent, retrieved_chunks, mcp_context
    """
    global _db_initialized

    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "mcp_context": {},
        "crew_result": "",
        "final_answer": "",
        "chat_history": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Running supervisor pipeline: thread={thread_id}, query='{query[:50]}...'")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL is not set, falling back to MemorySaver checkpointer")
        compiled_graph = _build_graph().compile(checkpointer=_memory)
        try:
            final_state = compiled_graph.invoke(initial_state, config)
        except Exception as e:
            logger.error(f"Supervisor pipeline failed (MemorySaver): {e}", exc_info=True)
            return {
                "final_answer": f"⚠️ An error occurred while processing your query: {str(e)}",
                "intent": "error",
                "retrieved_chunks": [],
                "mcp_context": {},
            }
    else:
        from langgraph.checkpoint.postgres import PostgresSaver
        try:
            with PostgresSaver.from_conn_string(db_url) as checkpointer:
                if not _db_initialized:
                    logger.info("Initializing Postgres checkpointer database tables...")
                    checkpointer.setup()
                    _db_initialized = True

                compiled_graph = _build_graph().compile(checkpointer=checkpointer)
                final_state = compiled_graph.invoke(initial_state, config)
        except Exception as e:
            logger.error(f"Supervisor pipeline failed (PostgresSaver): {e}", exc_info=True)
            return {
                "final_answer": f"⚠️ An error occurred while processing your query: {str(e)}",
                "intent": "error",
                "retrieved_chunks": [],
                "mcp_context": {},
            }

    return {
        "final_answer": final_state.get("final_answer", ""),
        "intent": final_state.get("intent", ""),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "mcp_context": final_state.get("mcp_context", {}),
    }

