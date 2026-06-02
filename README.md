# Intelligent Customer Service Agent

This project implements the PDF specification with a `LangGraph + ReAct + MySQL` customer service agent.
<img width="1028" height="990" alt="image" src="https://github.com/user-attachments/assets/7080ef7b-bc98-4305-aa4e-7db669d12467" />

<img width="1043" height="537" alt="image" src="https://github.com/user-attachments/assets/d3a1b751-e00a-4d05-9d44-5a9df7355167" />



## Requirements

- Docker Engine
- Docker Compose plugin
- Conda

## Features

- Tool-call planning for order tracking, profile lookup, refunds, complaints, and memory actions
- Structured planning with explicit order IDs, missing-slot verification, and bounded replanning before writes
- ReAct-style workflow with conditional read tools before verifier/action execution
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

The default demo backend is Google AI Studio with Gemma:

```bash
LLM_BACKEND=google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemma-4-31b-it
```

OpenAI remains supported:

```bash
LLM_BACKEND=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Tests

```bash
pytest
```

Run the natural-language PDF scorecard against the configured LLM and MySQL:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py
```

The scorecard reuses one `thread_id` across the PDF queries so `Cancel it` exercises short-term memory from the previous order query. It may write approved refunds and durable memory records during the run, then cleans up the demo database before exiting.

For a repeatable demo with the seeded MySQL rows, reset the known demo records first:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --reset-demo-data
```

To inspect the written rows after a run, opt out of cleanup:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --keep-demo-data
```

To show how each LangGraph node updates state during the run:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --reset-demo-data --show-node-trace
```

To rerun only selected scorecard cases:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --reset-demo-data --case 4 --case 5 --show-node-trace
```

The runtime graph uses model-generated tool calls with guarded write actions:

```text
planner -> read_tools -> planner/verifier -> actions/respond
```

The planner calls tools such as `order_lookup` and action proposals such as `propose_refund`. Read tools run only when the plan includes read calls; otherwise the plan goes directly to the verifier. The verifier checks database truth, required slots, explicit order IDs, and business rules, and only `proceed_to_action` proposals mutate MySQL.

This customer-service agent avoids hardcoded customer-facing response mappings. Instead, it uses structured planning, missing-slot verification, and bounded replanning to resolve ambiguous or incomplete information before executing customer-impacting actions.

## Project Structure

- `src/customer_service_agent/`: application code
- `sql/`: schema and seed data
- `compose.yaml`: Docker MySQL service
- `tests/`: functional tests for the scoring checklist
