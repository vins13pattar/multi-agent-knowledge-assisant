"""
crew.py — CrewAI Specialist Agents + Task Assembly.

Defines four role-based agents (Researcher, Summarizer, Analyst, Citation)
that collaborate in a sequential pipeline to process user queries
using RAG-retrieved context.
"""

import os
import logging
from typing import Optional

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  CrewAI Custom Tools
# ═══════════════════════════════════════════════════════════════════════════

@tool("rag_search_tool")
def rag_search_tool(query: str) -> str:
    """
    Search the document knowledge base for relevant context.
    Returns the top-5 most relevant text chunks with their sources.
    """
    from app.rag import retrieve_context
    chunks = retrieve_context(query, top_k=5)
    if not chunks:
        return "No relevant documents found in the knowledge base."

    results = []
    for chunk in chunks:
        results.append(
            f"[Source: {chunk.source} · Chunk {chunk.chunk_index} · "
            f"Score: {chunk.similarity_score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(results)


@tool("metadata_lookup_tool")
def metadata_lookup_tool(claim: str) -> str:
    """
    Look up source metadata for a factual claim to generate citations.
    Returns source document references with chunk IDs for citation.
    """
    from app.rag import retrieve_context
    chunks = retrieve_context(claim, top_k=3)
    if not chunks:
        return "No source found for this claim."

    citations = []
    for chunk in chunks:
        citations.append(
            f"[{chunk.source} · {chunk.chunk_index}] "
            f"(score: {chunk.similarity_score:.3f}): "
            f"{chunk.text[:200]}..."
        )
    return "\n".join(citations)


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Definitions
# ═══════════════════════════════════════════════════════════════════════════

def _create_agents() -> dict[str, Agent]:
    """Create the four specialist CrewAI agents."""

    llm_model = os.getenv("CREWAI_LLM_MODEL", "gpt-4o-mini")

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Retrieve accurate, comprehensive context from the document knowledge base",
        backstory=(
            "You have 15 years of experience in information retrieval and "
            "academic research. You excel at formulating precise search queries "
            "and identifying the most relevant passages from large document "
            "collections. You always use the rag_search_tool to find evidence "
            "before making any claims."
        ),
        tools=[rag_search_tool],
        llm=llm_model,
        verbose=False,
    )

    summarizer = Agent(
        role="Content Summarizer",
        goal="Synthesize retrieved context into clear, structured summaries",
        backstory=(
            "You are an award-winning technical writer who transforms complex, "
            "multi-source information into concise, well-organized summaries. "
            "You produce both bullet-point and narrative formats, always "
            "preserving key facts, figures, and nuances from the source material."
        ),
        llm=llm_model,
        verbose=False,
    )

    analyst = Agent(
        role="Domain Expert",
        goal="Answer questions accurately using only provided context, never hallucinate",
        backstory=(
            "You are a meticulous domain expert who provides precise, "
            "evidence-based answers. You NEVER make claims beyond what the "
            "provided context supports. If the context is insufficient, you "
            "clearly state what information is missing rather than guessing."
        ),
        llm=llm_model,
        verbose=False,
    )

    citation_agent = Agent(
        role="Citation Specialist",
        goal="Attach source references [filename·chunk_id] to every factual claim",
        backstory=(
            "You are a rigorous fact-checker and citation expert. Your job is "
            "to ensure every factual claim in the response is backed by a "
            "specific source reference in the format [filename · chunk_index]. "
            "You use the metadata_lookup_tool to verify and locate sources."
        ),
        tools=[metadata_lookup_tool],
        llm=llm_model,
        verbose=False,
    )

    return {
        "researcher": researcher,
        "summarizer": summarizer,
        "analyst": analyst,
        "citation_agent": citation_agent,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Task + Crew Assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_crew(
    query: str,
    context: list[RetrievedChunk],
    intent: str = "qa",
) -> str:
    """
    Assemble and execute the CrewAI specialist crew.

    Args:
        query: The user's question or request.
        context: List of retrieved chunks from RAG pipeline.
        intent: Query intent (research | summarize | qa | cite).

    Returns:
        Final response string from the crew pipeline.
    """
    agents = _create_agents()

    # Format context for agent consumption
    context_str = "\n\n".join([
        f"[Source: {c.source} · Chunk {c.chunk_index} · Score: {c.similarity_score:.3f}]\n{c.text}"
        for c in context
    ]) if context else "No context available."

    # ── Task 1: Research / Retrieve ─────────────────────────────────────
    research_task = Task(
        description=(
            f"Given the user query: '{query}'\n\n"
            f"And the following pre-retrieved context:\n{context_str}\n\n"
            "Review the provided context. If it seems insufficient, use the "
            "rag_search_tool to find additional relevant information. "
            "Compile all relevant findings into a structured set of key points."
        ),
        agent=agents["researcher"],
        expected_output="A structured list of key findings with source references.",
    )

    # ── Task 2: Summarize / Analyze ─────────────────────────────────────
    if intent == "summarize":
        analysis_description = (
            "Based on the research findings above, create a comprehensive "
            "structured summary with:\n"
            "- An executive summary paragraph\n"
            "- Key points as bullet items\n"
            "- Important figures, dates, and entities extracted\n"
            "Organize by theme or section if the content spans multiple topics."
        )
    else:
        analysis_description = (
            f"Based on the research findings above, answer the user's question: "
            f"'{query}'\n\n"
            "Provide a clear, accurate, evidence-based response. "
            "Only use information from the provided context. "
            "If the context is insufficient, clearly state what's missing."
        )

    analysis_task = Task(
        description=analysis_description,
        agent=agents["summarizer"] if intent == "summarize" else agents["analyst"],
        expected_output=(
            "A clear, well-structured response that directly addresses the query."
        ),
    )

    # ── Task 3: Citation ────────────────────────────────────────────────
    citation_task = Task(
        description=(
            "Review the response above and ensure every factual claim has a "
            "proper citation in the format [filename · chunk_index]. \n\n"
            "Rules:\n"
            "1. Every factual statement must have at least one citation.\n"
            "2. Use the metadata_lookup_tool to verify source references.\n"
            "3. Format: [source_name · chunk_number]\n"
            "4. Place citations inline, immediately after the claim they support.\n"
            "5. At the end, include a 'Sources' section listing all cited documents."
        ),
        agent=agents["citation_agent"],
        expected_output=(
            "The complete response with inline citations [source · chunk_id] "
            "on every factual claim, plus a Sources section at the end."
        ),
    )

    # ── Assemble and Execute Crew ───────────────────────────────────────
    crew = Crew(
        agents=[agents["researcher"], agents["summarizer"], agents["analyst"], agents["citation_agent"]],
        tasks=[research_task, analysis_task, citation_task],
        process=Process.sequential,
        verbose=False,
    )

    logger.info(f"Kicking off crew for intent='{intent}', query='{query[:50]}...'")

    try:
        result = crew.kickoff()
        return result.raw if hasattr(result, "raw") else str(result)
    except Exception as e:
        logger.error(f"Crew execution failed: {e}", exc_info=True)
        raise
