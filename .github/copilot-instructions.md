# Copilot instructions

The full guidance for coding agents is in [`AGENTS.md`](../AGENTS.md) at the repository root. Read it before making changes. The rules that matter most:

- **Use uv for everything:** `uv sync`, `uv add`, `uv run pytest`, `uv run ruff check`. Never use `pip install`.
- **Never call a paid service unless asked.** The notebook and `generate-data` (without `--dry-run`) call Azure AI Foundry and TypeSafe AI's Jev.
- **Tests must never call a live service.** Stub Foundry and Jev as the existing tests in `tests/unit/` do.
- **Never delete the caches** in `data/`, and never commit `.env` or anything under `data/`.
- **Do not run `ruff format`.** The code is hand-formatted; `uv run ruff check` must pass.
- **Logic lives in `src/`.** The notebook only calls functions, and `app/streamlit_app.py` is layout and state only.
- **Bump `QUESTION_SET_VERSION`** in `src/jev_classifier/questions.py` whenever a Jev question changes.
- **Keep the docs true:** `README.md`, `docs/architecture.md`, `docs/requirements.md` and `docs/data-model.md`.
