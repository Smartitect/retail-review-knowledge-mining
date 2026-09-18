# retail-review-knowledge-mining
A demo of various techniques for mining insights from semi structured data.

## Demo: classifying BBQ reviews with Jev

`data/input/product_reviews.json` holds 256 fictional reviews of 17 barbecue products (copied from
`endjin/endjin-text-analytics-streamlit`). [`notebooks/01_classify_reviews_with_jev.ipynb`](notebooks/01_classify_reviews_with_jev.ipynb)
loads them into a flat Polars frame, splits them into sentences, and asks
[TypeSafe AI's Jev](https://docs.typesafe.ai/introduction) about every sentence, with the whole review as context:

- **frustration** (Score 0–4), **problem category** (Choice), **products mentioned** (one Noul per catalogue product), **language** (Choice)
- plus sentiment, recommendation, churn risk, safety concern, improvement suggestion and competitor mention

The rating and demographics are not sent, so frustration vs star rating is a fair check of the model.

| Package | Responsibility |
|---|---|
| `src/review_wrangler` | Load and flatten the JSON; split reviews into sentences |
| `src/jev_classifier` | The question set, and async classification with a Parquet cache |
| `src/review_insights` | Review and customer rollups, Sankey flows, risk tiers, escalation and review queues |
| `src/review_charts` | Plotly figures and endjin colour roles for the dashboard |
| `app/streamlit_app.py` | The dashboard: layout and state only |

### Dashboard

```powershell
uv run streamlit run app/streamlit_app.py
```

Reads `data/output/sentences_classified.parquet`, so run the notebook first. One review is one customer.

- **Sentiment flow** - a Sankey from customer sentiment through product category and product to each customer's
  *primary issue* (the problem in their most frustrated sentence), so every customer is exactly one path and widths
  are customer counts. Sentiment can come from the text (Jev) or the star rating; a drill-down lists the customers
  behind any issue.
- **Customers at risk** - tenure against lifetime revenue (log scale), shaped and coloured by risk tier. Box or lasso
  select to list customers. Risk thresholds are set in the sidebar.

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
