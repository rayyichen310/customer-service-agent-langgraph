# Intelligent Customer Service Agent

A LangGraph-powered customer-service agent that plans tool calls, verifies customer-impacting actions, and uses MySQL-backed memory to handle realistic order, refund, complaint, and personalization workflows.

This is not a hardcoded response map. The agent uses model-generated tool calls, database reads, verifier-controlled policy decisions, and bounded replanning before any customer-impacting write.

<img width="1043" height="537" alt="Customer service agent demo console" src="https://github.com/user-attachments/assets/d3a1b751-e00a-4d05-9d44-5a9df7355167" />

## What This Demonstrates

- Natural-language order lookup, refund, cancellation, complaint, and preference workflows
- Guarded write actions through verifier-controlled policy gates
- Short-term thread memory through the LangGraph checkpointer
- Long-term customer memory and complaint history stored in MySQL
- A realtime React console that streams and visualizes each graph node update
- A runnable self-evaluation scorecard aligned with the submitted behavior table

## Quickstart

### Requirements

- Docker Engine
- Docker Compose plugin
- Conda
- Node.js and npm for the local demo console

### 1. Create the Python Environment

```bash
conda env create -f environment.yml
conda activate customer-service-agent
pip install -e ".[dev]"
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Fill in `GOOGLE_API_KEY` in `.env`. The example file already defaults to the Google AI Studio backend with Gemma and development-only demo endpoints:

```env
APP_ENV=development
LLM_BACKEND=google
GOOGLE_MODEL=gemma-4-31b-it
```

### 3. Start MySQL

```bash
docker compose up -d
```

The container loads `sql/schema.sql` and `sql/seed.sql` automatically on first startup.

### 4. Start the API

```bash
customer-agent-api
```

The FastAPI server listens on `http://127.0.0.1:8000`.

### 5. Start the Realtime Demo Console

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed by the terminal, usually `http://127.0.0.1:5173`. The console proxies `/api/*` to the FastAPI server on `http://127.0.0.1:8000`.

Use the `Initialize` button before a repeatable demo. It resets the seeded demo rows through `POST /demo/reset`, which is available only when `APP_ENV=development`.

## Demo Console

The local console streams `/chat/stream` events from FastAPI and visualizes the request as it moves through planning, read tools, verification, action execution, and response generation.

<img width="1028" height="990" alt="LangGraph trace and database inspector" src="https://github.com/user-attachments/assets/7080ef7b-bc98-4305-aa4e-7db669d12467" />

Representative prompts:

- `Where is my order 12345?`
- `Refund order 7890 if delivered`
- `Cancel it`
- `Remember I prefer refunds`
- `My order is late again`
- `Refund order 0000`

Demo customers:

| Customer | Seeded Orders |
| --- | --- |
| Alice Chen, customer `1` | `12345`, `1001`, `7890` |
| Bob Lin, customer `2` | `5678` |
| Carol Wang, customer `3` | `2222` |

## Architecture

```text
User message
  -> planner
  -> read_tools
  -> verifier
  -> actions or respond
```

| Component | Responsibility |
| --- | --- |
| `planner` | Produces read tool calls and proposed customer-impacting actions. |
| `read_tools` | Reads orders, customer profile, customer memory, issue history, and order lists. |
| `verifier` | Enforces identity, required slots, order ownership, order status, and write-safety rules. |
| `actions` | Executes only verifier-approved writes. |
| `respond` | Generates the final answer from verified facts, verification decisions, and memory. |

Read tools run only when the plan includes read calls. The verifier can ask for missing information, block unsafe actions, proceed to a write, or send planner feedback for bounded replanning.

## Safety Model

The LLM does not directly mutate the database. It proposes actions such as `propose_refund`, `propose_cancel_order`, `propose_log_complaint`, or `propose_write_memory`; the verifier checks identity, required slots, order ownership, order status, and memory-write rules before the `actions` node can write to MySQL.

Examples:

- Refunds require an order lookup and are allowed only for delivered orders.
- Cancellation is blocked for delivered or refund-requested orders.
- Complaints require an issue description before they are logged.
- Invalid order IDs are blocked with verifier policy errors such as `ORDER_NOT_FOUND`.

## Evaluation

The self-evaluation scorecard runs the behavior checklist as natural-language requests against the real agent, database reads, verifier, and write paths. For the full submitted table, see [self_evaluation_table.pdf](self_evaluation_table.pdf).

| # | Test Query | Expected Behavior |
| --- | --- | --- |
| 1 | `Where is my order 12345?` | Uses `order_lookup` and returns the `in_transit` status. |
| 2 | `Check status of order 1001.` | Fetches the requested order and reports `processing`. |
| 3 | `Show my profile.` | Uses `customer_profile` and returns the authenticated customer's name and email. |
| 4 | `Refund order 5678.` | Looks up the order, passes verifier checks, executes `propose_refund`, and confirms the refund request. |
| 5 | `I want to complain about order 2222.` | Detects the missing issue description and asks for details instead of logging an empty complaint. |
| 6 | `Refund order 7890 if delivered.` | Looks up the order, verifies that it is delivered, then executes the refund request. |
| 7 | `Cancel it.` | Uses short-term memory to resolve `it`, then blocks cancellation because a refund was already requested. |
| 8 | `What issues have I had before?` | Reads customer issue history and summarizes prior complaints and patterns. |
| 9 | `Remember I prefer refunds.` | Writes a durable refund preference to customer memory after verifier approval. |
| 10 | `My order is late again.` | Combines short-term and long-term context for a personalized response without logging an ambiguous complaint. |
| 11 | `Refund order 0000.` | Blocks the refund mutation because the verifier detects `ORDER_NOT_FOUND`. |

Run the scorecard:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --reset-demo-data --show-node-trace
```

The scorecard exercises real write paths such as refund requests and memory writes. By default it resets demo rows before exit; use `--keep-demo-data` only when you want to inspect the written rows afterward.

Run only selected cases:

```bash
conda run --no-capture-output -n customer-service-agent python scripts/run_scorecard.py --reset-demo-data --case 4 --case 5 --show-node-trace
```

## Other Interfaces

### CLI

Interactive chat:

```bash
customer-agent --interactive --customer-id 1
```

Single query:

```bash
customer-agent "Where is my order 12345?" --customer-id 1
```

### REST API

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","customer_id":1,"message":"Refund order 7890 if delivered"}'
```

### Streaming API

`POST /chat/stream` emits Server-Sent Events:

- `started`
- `node`
- `final`
- `error`

The React console uses this endpoint to render the live LangGraph trace.

Demo reset and database snapshot endpoints are development-only and require `APP_ENV=development`.

## Alternative LLM Backend

Google AI Studio with Gemma is the default demo path. OpenAI is also supported by changing `.env`:

```env
LLM_BACKEND=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4.1-mini
```

## Tests and Build

Backend tests:

```bash
conda run --no-capture-output -n customer-service-agent pytest
```

Frontend build:

```bash
cd frontend
npm run build
```

## Stop Services

```bash
docker compose down
```

## Project Structure

- `src/customer_service_agent/graph/agent.py`: LangGraph state machine
- `src/customer_service_agent/graph/policy.py`: verifier and write-safety rules
- `src/customer_service_agent/graph/actions.py`: approved MySQL mutations
- `src/customer_service_agent/graph/response_facts.py`: facts used to ground final responses
- `src/customer_service_agent/repository.py`: MySQL reads and writes
- `src/customer_service_agent/api.py`: FastAPI REST and streaming endpoints
- `scripts/run_scorecard.py`: self-evaluation runner
- `frontend/`: realtime React demo console
- `sql/`: schema and seed data
