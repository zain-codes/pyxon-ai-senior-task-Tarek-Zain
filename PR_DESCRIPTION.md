## Summary

Implemented all three agent types plus bonus deliverables: multi-agent swarm with supervisor pattern, RAG integration with FAISS, comprehensive test suite (15 tests), Docker containerization, and Kubernetes Helm chart.

## Contact Information

📧 Email: zain20tarek@gmail.com
📱 Phone: +966 538456188

## Features Implemented

- [x] Agent with search data source (Tavily) + LLM answers
- [x] Agent that requests URLs/APIs and uses content to answer
- [x] Agent swarm (multi-agent) with LangGraph supervisor
- [x] Full end-to-end example (question → data → answer)
- [x] README with run instructions and architecture
- [x] (Optional) RAG integration with FAISS vector store
- [x] (Optional) Tests with mocked external calls (15 cases, zero API keys needed)
- [x] (Optional) Docker + Docker Compose
- [x] (Optional) Kubernetes Helm chart outline

## Architecture

Supervisor pattern using `langgraph-supervisor`. The supervisor routes queries to:

1. **research_agent** — Tavily web search for general questions and current events
2. **url_agent** — httpx fetch + trafilatura/BeautifulSoup for specific URLs and APIs
3. **synthesizer_agent** — tool-less writer that produces the final polished answer

Optional RAG layer indexes agent findings in a FAISS vector store (HuggingFace `all-MiniLM-L6-v2` embeddings) before the synthesis step, retrieving only the most relevant chunks to improve grounding.

See `README.md` for the full Mermaid architecture diagram.

## How to Run

```bash
git clone https://github.com/zain-codes/pyxon-ai-senior-task-Tarek-Zain.git
cd pyxon-ai-senior-task-Tarek-Zain
cp .env.local.example .env.local     # fill in AWS credentials (provided separately)
python setup_env.py                  # fetches API keys from AWS Secrets Manager → writes .env
docker compose build
docker compose run agent python main.py
docker compose run agent python main.py --interactive
```

## Example Questions & Behavior

1. **"What are the latest developments in AI regulation?"** → `research_agent` searches Tavily for current articles → `synthesizer_agent` writes a structured answer with `[Source Title](URL)` citations.
2. **"What does https://jsonplaceholder.typicode.com/posts/1 return?"** → `url_agent` fetches the JSON API → `synthesizer_agent` explains the data structure and fields.
3. **"What is LangGraph and how does it compare to other agent frameworks?"** → supervisor routes to `research_agent` → Tavily search → RAG indexes findings in FAISS → `synthesizer_agent` produces a grounded answer with citations.

## Assumptions

- **LLM**: Groq free tier — Llama 3.3 70B Versatile (fastest free inference)
- **Search**: Tavily free tier (1,000 credits/month, basic depth = 1 credit/query)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, CPU, ~80 MB download on first use)
- **Vector Store**: FAISS (local, no server required)
- **Python**: 3.11+
- **Testing**: All 15 tests pass with zero API keys (fully mocked)
