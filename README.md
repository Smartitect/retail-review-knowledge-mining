# retail-review-knowledge-mining
A demo of various techniques for mining insights from semi structured data.

## Demo: classifying BBQ reviews with Jev

A fictional barbecue retailer, generated deterministically: customers (Faker), their orders over time (with
seasonality and realistic buying patterns), and the reviews a biased subset of them chose to write. See
[`docs/data-model.md`](docs/data-model.md). [`notebooks/01_classify_reviews_with_jev.ipynb`](notebooks/01_classify_reviews_with_jev.ipynb)
generates the dataset, loads one row per review with point-in-time customer features, splits reviews into sentences,
and asks [TypeSafe AI's Jev](https://docs.typesafe.ai/introduction) about every sentence, with the whole review as context:

- **frustration** (Score 0–4), **problem category** (Choice), **products mentioned** (one Noul per catalogue product), **language** (Choice)
- plus sentiment, recommendation, churn risk, safety concern, improvement suggestion and competitor mention

The rating and customer facts are not sent, so frustration vs star rating is a fair check of the model.

Review text comes from a chat deployment on **Azure AI Foundry** (set `AZURE_FOUNDRY_*` in `.env` and
`WRITER = "foundry"` in the notebook), or, offline, from the 256 hand-written reviews in
`data/input/product_reviews.json` (copied from `endjin/endjin-text-analytics-streamlit`), labelled `template`.

| Package | Responsibility |
|---|---|
| `src/retail_model` | Table schemas (pandera), cross-table integrity, review propensity, dataset storage |
| `src/retail_generator` | Deterministic customers, orders and reviews; every knob in `GeneratorConfig` |
| `src/review_writer` | Review text from Azure AI Foundry or templates, cached by prompt |
| `src/customer_features` | Point-in-time features: only data from before each review, never after |
| `src/review_wrangler` | Load reviews with their text, product and features; split into sentences |
| `src/jev_classifier` | The question set, and async classification with a Parquet cache |
| `src/review_insights` | Review and customer views, Sankey flows, risk tiers, escalation and review queues |
| `src/review_charts` | Plotly figures and endjin colour roles for the dashboard |
| `app/streamlit_app.py` | The dashboard: layout and state only |

### Scaling the dataset

The dataset grows in batches with `generate-data`, so you choose how much compute, storage and API spend to take on:

```powershell
uv run generate-data init --seed 42 --as-of 2026-06-30
uv run generate-data add-customers --count 1000 --dry-run --foundry-input-price 0.15 --foundry-output-price 0.60
uv run generate-data add-customers --count 1000            # review text from Azure AI Foundry
uv run generate-data add-customers --total 5000 --writer template
uv run generate-data advance --to 2026-12-31               # existing customers keep buying and reviewing
uv run generate-data fill-texts                            # retry any review text that failed
uv run generate-data status
```

- **Batches** are files: `data/generated/<table>/batch-NNNN.parquet`, plus `manifest.json`. A batch is atomic: the manifest is written last, and readers only see batches it lists.
- **Deterministic:** the same seed gives the same data, and two batches of 1,000 customers equal one batch of 2,000. Advancing time gives exactly what generating to the later date would have.
- **Idempotent:** `--total` and `advance --to` do nothing when the dataset is already there.
- **Explicit about text:** Foundry is the default writer and refuses to run unconfigured; `--writer template` must be chosen explicitly.
- **Costed first:** every batch reports what its reviews will cost to write and to classify, and `--dry-run` reports without writing. Jev only pays for sentences it has not seen: the notebook classifies from a cache, and template text is almost all cached already.

Measured on the defaults (1,000 and 10,000 customers; 100,000 extrapolated). Jev is at ~2,750 input tokens and ~0.3 s per sentence, $0.042 per million tokens, concurrency 8. Foundry times assume ~2.5 s per review at concurrency 8, and its cost depends on the deployment's prices.

| Customers | Orders | Reviews | Sentences | Generate | Storage | Foundry tokens (in / out) | Foundry time | Jev cost | Jev time |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | ~8,400 | ~1,500 | ~6,800 | ~2 s | ~0.4 MB | 0.4M / 0.2M | ~8 min | ~$0.79 | ~4 min |
| 10,000 | ~76,000 | ~13,700 | ~61,000 | ~15 s | ~3 MB | 3.6M / 1.7M | ~70 min | ~$7 | ~40 min |
| 100,000 | ~760,000 | ~137,000 | ~610,000 | ~2.5 min | ~30 MB | 36M / 17M | ~12 h | ~$71 | ~8.5 h |

At 100,000 customers Jev's published limit of 1,200 requests a minute, not concurrency, sets the pace. Grow in batches, and classify each before adding the next.

### Dashboard

```powershell
uv run streamlit run app/streamlit_app.py
```

Reads `data/output/sentences_classified.parquet`, so run the notebook first.

- **Sentiment flow** - a Sankey from review sentiment through product category and product to each review's
  *primary issue* (the problem in its most frustrated sentence), so every review is exactly one path and widths
  are review counts. Sentiment can come from the text (Jev) or the star rating; a drill-down lists the reviews
  behind any issue.
- **Customers at risk** - one mark per customer at their most recent review: tenure against lifetime revenue as at
  that review (log scale), shaped and coloured by risk tier. Box or lasso select to list customers. Risk thresholds
  are set in the sidebar.

Colours are endjin's (read from endjin.com), assigned by role and checked for colour-blind separation; see
`src/review_charts/palette.py`. They are validated for the light theme, which `.streamlit/config.toml` pins.

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY`. Answers are cached in `data/output/` (gitignored)
under the question-set version, so changing a question in `questions.py` means bumping `QUESTION_SET_VERSION`.
The unit tests stub the client and never call the live API.

To see the JSON sent to and received from Jev, call `jev_classifier.transcript.log_to_stdout()` before classifying:
each exchange prints as one JSON object with the request and response bodies exactly as they crossed the wire, the
request ID and latency. The question set prints in full once per run (`full_questions=True` prints it every time), and
the API key is never printed. It is off by default; `transcript.silence()` turns it off again.
Two real exchanges captured this way, with the questions in full:
[`docs/jev-example-positive.json`](docs/jev-example-positive.json) ("Solid grill cover that fits my FireMaster
perfectly.") and [`docs/jev-example-negative.json`](docs/jev-example-negative.json) ("Waste of money - get a different
brand.").

## Getting started

Open the repository in VS Code and choose **Reopen in Container**. The dev container
provides PowerShell (the default terminal), the GitHub CLI, and uv — which owns the
Python interpreter. `.devcontainer/postCreateCommand.ps1` installs uv, runs
`uv sync --python 3.12`, and generates a PowerShell profile that activates `.venv`
automatically in every terminal.

### Working with dependencies

```powershell
uv add <package>            # runtime dependency
uv add --dev <package>      # tooling
uv sync                     # after a pull that changes pyproject.toml or uv.lock
uv run pytest               # run inside the synced environment
```

Never `pip install`, and never activate the virtualenv by hand — `uv run` syncs first.
`pyproject.toml` is the only place dependencies are declared, and `uv.lock` is committed.

### Smoke test after a container rebuild

```powershell
uv --version
python --version   # must report 3.12 from .venv, not a system interpreter
gh --version
pytest --version
```

Rebuild *without cache* periodically: a definition that only works incrementally is
broken for the next person who clones it.
