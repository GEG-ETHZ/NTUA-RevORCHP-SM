# Tests

The tests check that every command example in the main `README.md` and a set
of direct Python calls still give the stored reference outputs in `data/`.
From the repository directory (Windows PowerShell shown; on Linux or macOS use
`.venv/bin/python`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

They also run automatically on GitHub (Linux and Windows) for every pull
request. A failure means an output changed; read `generate_reference.py`
before regenerating the reference files.
