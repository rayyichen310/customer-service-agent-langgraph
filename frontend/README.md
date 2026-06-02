# Customer Service Agent Console

Local demo console for the existing FastAPI customer-service agent.

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to the existing FastAPI server on `http://127.0.0.1:8000`.

The Initialize button calls `POST /api/demo/reset`, which is provided by the Python API in development mode and runs `frontend/dev/reset_demo_data.py`.
