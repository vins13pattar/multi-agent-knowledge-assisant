# 🧠 Multi-Agent Knowledge Assistant

An enterprise-grade RAG + multi-agent system that ingests documents, orchestrates specialized agents via CrewAI and LangGraph, and exposes a Streamlit UI — fully Dockerized.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Docker Compose                      │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           App Container                       │   │
│  │                                               │   │
│  │  ┌─────────────────┐  ┌──────────────────┐   │   │
│  │  │  Streamlit :8501 │  │ MCP Server :8100 │   │   │
│  │  │                  │  │                  │   │   │
│  │  │  Chat UI         │  │  Context Store   │   │   │
│  │  │  LangGraph       │──│  Message Bus     │   │   │
│  │  │  CrewAI Agents   │  │  Embedding Store │   │   │
│  │  │  RAG Pipeline    │  │                  │   │   │
│  │  └─────────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────┘   │
│                         │                            │
│                         ▼                            │
│  ┌──────────────────────────────────────────────┐   │
│  │          ChromaDB Container :8000             │   │
│  │          Vector Store (persistent)            │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangGraph (StateGraph + MemorySaver) |
| Multi-Agent | CrewAI (4 specialist agents) |
| Protocol | MCP SDK (HTTP/SSE transport) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Frontend | Streamlit |
| Validation | Pydantic v2 + jsonschema |
| Containerization | Docker + docker-compose |
| Language | Python 3.11 |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- (Optional) Firecrawl API Key for URL ingestion
- (Optional) LangSmith API Key for tracing

### 1. Clone & Configure

```bash
cd multi-agent-knowledge-assistant
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run with Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

### 3. Open the App

Navigate to [http://localhost:8501](http://localhost:8501)

### 4. Demo Run

1. Upload a PDF or TXT document via the sidebar
2. (Optional) Ingest a URL using Firecrawl
3. Ask questions like:
   - "What is this document about?"
   - "Summarize the key points"
   - "What are the citations for the main claims?"

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for GPT-4o-mini |
| `CHROMA_HOST` | ✅ | ChromaDB host (set to `chroma` in Docker) |
| `CHROMA_PORT` | ✅ | ChromaDB port (default: 8000) |
| `MCP_SERVER_URL` | ✅ | MCP server URL (default: http://localhost:8100) |
| `FIRECRAWL_API_KEY` | ❌ | Firecrawl API key for URL ingestion |
| `LANGSMITH_TRACING` | ❌ | Enable LangSmith tracing (true/false) |
| `LANGSMITH_API_KEY` | ❌ | LangSmith API key |
| `LANGSMITH_PROJECT` | ❌ | LangSmith project name |

## Project Structure

```
multi-agent-knowledge-assistant/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── supervisor.py        # LangGraph state machine
│   ├── crew.py              # CrewAI agents + tasks
│   ├── rag.py               # Ingestion + retrieval pipeline
│   ├── mcp_client.py        # MCP HTTP client
│   └── models.py            # Pydantic v2 schemas
├── mcp_server/
│   └── server.py            # FastMCP server (HTTP/SSE)
├── docker/
│   ├── Dockerfile           # App + MCP server container
│   ├── docker-compose.yml   # 2-container orchestration
│   └── supervisord.conf     # Process manager
├── .env.example
├── requirements.txt
└── README.md
```

## Agent Pipeline

```
User Query → [Classify Intent] → [Retrieve Context] → [Build MCP Context]
           → [CrewAI Agents] → [Format Response] → User
```

### Agents

| Agent | Role | Purpose |
|-------|------|---------|
| Researcher | Senior Research Analyst | Retrieves context from ChromaDB |
| Summarizer | Content Summarizer | Synthesizes multi-chunk summaries |
| Analyst | Domain Expert | Answers questions with evidence |
| Citation Agent | Citation Specialist | Attaches [source · chunk_id] references |

## License

MicroDegree Capstone Project — Part 3: Deployment
