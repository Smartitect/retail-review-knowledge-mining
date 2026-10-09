# Agent instructions

Guidance for AI coding agents (GitHub Copilot, Claude Code and others) working in this repository. People should start with the [README](README.md).

## What this is

A demo of mining insight from semi-structured customer feedback. A deterministic generator builds a fictional barbecue retailer (customers, orders and reviews). Azure AI Foundry writes the review text. TypeSafe AI's Jev classifies every sentence, and a notebook and a Streamlit dashboard turn the answers into product and customer insight.

Read these before changing anything substantial:

- [`docs/architecture.md`](docs/architecture.md): the pipeline, the packages and their dependencies, where data lives, and how to extend the demo.
- [`docs/requirements.md`](docs/requirements.md): what the system must do, with the test that verifies each requirement.
- [`docs/data-model.md`](docs/data-model.md): the tables, keys, integrity rules and generator behaviour.

## Commands

The shell is PowerShell in the dev container, but every command below also works in bash.

```powershell
uv sync                                         # install or update the environment from uv.lock
uv run pytest                                   # unit tests: offline, about 10 seconds
uv run ruff check                               # lint: must pass before any PR
uv run generate-data --help                     # the dataset CLI
uv run streamlit run app/streamlit_app.py       # the dashboard
```

## Rules

### Environment

- **Use uv for everything.** `uv add <pkg>` for runtime dependencies, `uv add --dev <pkg>` for tooling, and `uv run <cmd>` to run anything. Never use `pip install`, never edit `uv.lock` by hand, and never activate the virtualenv yourself.
- **Python 3.12.** Dependencies are declared only in `pyproject.toml`.
- **DSPy uses its `lm15` engine** (`dspy_classifier/foundry.py`). Its `litellm` engine does not import alongside `openai` 3.x, so never switch to it, and never `import litellm` in a process that has already imported `dspy`.
- **A new package under `src/` must be added** to `[tool.hatch.build.targets.wheel].packages` in `pyproject.toml`, or it will not import.

### Cost and secrets

- **Never call a paid service unless asked to.** `generate-data add-customers`, `advance` and `fill-texts` (without `--dry-run`) call Azure AI Foundry. Running either notebook calls both Foundry and Jev. Use `--dry-run` to see what a batch would cost.
- **Never delete or rewrite the caches** `data/generated/review_text_cache.parquet` or `data/output/jev_sentence_answers.parquet`. They hold work that was paid for.
- **Never commit** `.env`, anything under `data/`, or any key, endpoint or deployment name. The Foundry endpoint, key and deployment, and the Jev key, live only in `.env`; `.env.example` holds placeholders.
- **Never print a secret.** Keep `ModelSettings.__repr__` hiding the key, and never record request headers in `jev_classifier.transcript`.

### Tests

- **Tests must never call a live service.** Use `tests/unit/stub_writer.py` for review text, an `httpx2.MockTransport` for Foundry (see `tests/unit/test_review_writer.py`), and stub responses for Jev. Look at the existing tests for the pattern.
- **Name tests as statements of behaviour**, for example `test_two_batches_of_customers_equal_one`.
- **When you add or change a requirement, update `docs/requirements.md`** and name the test that verifies it.

### Invariants that must hold

These are what make the demo trustworthy. Each one has tests that will fail if you break it.

- **Determinism.** Every random draw comes from `retail_generator.random_streams.rng(seed, stream, key)`, one stream per entity. Never use a shared or global random generator, and never reorder draws within an existing stream: that changes every dataset already generated.
- **Batching never changes the data.** Two batches of customers equal one, and `advance` equals generating straight to the later date.
- **Point in time.** Every customer feature goes through `customer_features.point_in_time`, which only admits events strictly before the review.
- **Truth is never a feature.** `review_truth` is only for scoring classifiers.
- **Jev never sees the star rating or demographics** (`jev_classifier.questions.build_state`).
- **Failures are never cached or filled in**, for either Foundry or Jev.
- **Cache keys cover everything that changes the answer.** If you change a Jev question, bump `QUESTION_SET_VERSION` in `src/jev_classifier/questions.py`. If you change the review prompt, `prompt_hash` already covers it.
- **Every rate, weight and date for the generator lives in `GeneratorConfig`** (`src/retail_generator/config.py`).

### Where code goes

- **Logic lives in `src/`.** The notebook only calls functions; `app/streamlit_app.py` is layout and state only. Build frames in `review_insights` and figures in `review_charts`.
- **Respect the package dependencies** shown in `docs/architecture.md`. `retail_model`, `customer_features`, `jev_classifier`, `review_writer` and `review_charts` import no other project package. Do not add a cycle.
- **Validate at the boundaries.** New tables get a pandera `DataFrameModel` in `src/retail_model/tabular_schemas.py`, and cross-table rules go in `integrity.py`.

## Code style

- **Polars, not pandas.** Prefer expressions and `LazyFrame`s.
- **Timestamps are naive UTC** throughout the data model. `DTZ001` is ignored in `tests/` and `notebooks/` for that reason. In `src/`, mark a deliberate naive `datetime` with `# noqa: DTZ001` and a reason.
- **Configuration is frozen dataclasses**, as in `config.py` and `review_propensity.py`.
- **Do not run `ruff format`.** The code is hand-formatted to about 120 columns, and the formatter would rewrite most files. `uv run ruff check` must pass.
- **Module docstrings explain why.** Each module opens with a docstring saying what it is for and the decisions behind it. Keep it true when you change the module. Inline comments are rare and explain intent, not mechanics.
- **British English** in prose, docstrings and docs (colour, behaviour, catalogue).
- **Keys are strings with a type prefix:** `C0000001`, `P001`, `O000000001`, `L0000000001`, `R0000000001`.

## Pull requests

- Branch from `main`. One concern per PR.
- Before opening a PR, run `uv run pytest` and `uv run ruff check`. CI runs both.
- Write the description with **Why**, **What changed** and **Testing** sections, as the existing PRs do.
- Update the README and the docs in `docs/` when behaviour, commands or setup change.
- Diagrams are SVGs generated by `docs/diagrams/build_diagrams.py`, each with a light and a dark variant. See [`docs/diagrams/README.md`](docs/diagrams/README.md) before changing one, and never replace them with Mermaid.
