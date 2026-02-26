Face Verification React Frontend (Vite)
======================================

Commands
--------
- npm install
- npm run dev (dev server on http://localhost:5173 with proxy to FastAPI on 8000)
- npm run build (builds to dist/)
- npm run preview

Notes
-----
- Proxy configured in vite.config.js for /register, /verify, /health -> http://127.0.0.1:8000
- Ensure FastAPI is running on port 8000.
- Tailwind CSS already configured. Edit src/ui/App.jsx for UI logic.

Quick backend start
-------------------
- Create a Python venv and install requirements from the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate    # PowerShell/Windows
pip install -r ../requirements.txt
```

- Start the FastAPI server (from project root):

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend note
-------------
- The frontend reads `VITE_API_BASE` (env) to prepend API calls. In development this defaults to the Vite proxy. To target a specific API URL set `VITE_API_BASE` in a `.env` file or your shell.
