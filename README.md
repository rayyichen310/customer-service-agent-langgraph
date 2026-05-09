# Intelligent Customer Service Agent

This project implements the PDF specification with a `LangGraph + ReAct + MySQL` customer service agent.

## Requirements

- Docker Engine
- Docker Compose plugin
- Conda

## Features

- Intent parsing for order tracking, profile lookup, refunds, complaints, and memory actions
- ReAct-style workflow with `planner -> tools -> memory -> verifier -> response`
- Short-term memory through LangGraph checkpointer
- Long-term memory through MySQL tables: `customer_memory` and `complaints`
- FastAPI endpoint and CLI interface
- Seed data and tests mapped to the spec's scoring checklist

## Python Environment

```bash
conda env create -f environment.yml
conda activate customer-service-agent
pip install -e .
cp .env.example .env
```

## MySQL Setup With Docker

Verify that the Compose plugin is installed:

```bash
docker compose version
```

Start MySQL 8.4:

```bash
docker compose up -d
```

The container loads `sql/schema.sql` and `sql/seed.sql` automatically on first startup.

To stop it later:

```bash
docker compose down
```

## Run

CLI:

```bash
customer-agent --interactive --customer-id 1
```

Single query:

```bash
customer-agent "Where is my order 12345?"
```

API:

```bash
customer-agent-api
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","customer_id":1,"message":"Refund order 7890"}'
```

## LLM Mode

Default mode is `LLM_BACKEND=heuristic`, which works without API keys and is useful for grading and deterministic testing.

To use OpenAI:

```bash
LLM_BACKEND=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Tests

```bash
pytest
```

## Project Structure

- `src/customer_service_agent/`: application code
- `sql/`: schema and seed data
- `compose.yaml`: Docker MySQL service
- `tests/`: functional tests for the scoring checklist
