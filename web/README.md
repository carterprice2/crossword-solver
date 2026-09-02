# Web UI

Vite + React frontend for the crossword agent. Full setup is in the root
[README, "Run locally"](../README.md#run-locally).

From the **repo root**, with the `[web]` extra installed:

```bash
# one process: build dist/, serve API + UI
make serve                         # http://127.0.0.1:8000

# or hot reload: API in one terminal, Vite in another
make serve-dev                     # http://127.0.0.1:8000  (API)
cd web && npm run dev              # http://127.0.0.1:5173  (proxies /api to :8000)
```

Inside `web/`:

```bash
npm install
npm run dev      # http://127.0.0.1:5173  (proxies /api to :8000)
npm run build    # writes dist/, served by `python3 -m crossword serve --build`
```
