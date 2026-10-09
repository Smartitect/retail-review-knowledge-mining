# Retail review knowledge mining

A demo of mining actionable insight from semi-structured customer feedback: free-text product reviews, in several languages, tied to structured customer and order data.

It is built around a fictional barbecue retailer, so it can be shared, rerun and scaled without any real customer data:

1. **Generate** a realistic retail dataset: customers, their orders over time, and the reviews a biased subset of them chose to write. A chat model on Azure AI Foundry writes the review text, in English or the customer's own language.
2. **Classify** every review sentence with [TypeSafe AI's Jev](https://docs.typesafe.ai/introduction), which answers typed questions with probabilities rather than generating text: how frustrated is the customer, what kind of problem is it, which products are mentioned, is there a safety concern or a churn signal?
3. **Explore** the answers in a notebook and an interactive dashboard: which products have which problems, which customers are at risk, and how much revenue they represent.

Jev is never shown the star rating, so "does frustration track the stars?" is a fair test of the model. The generator records what each review was meant to say, so Jev's answers can be checked against the truth.

| Document | What it covers |
|---|---|
| [`docs/requirements.md`](docs/requirements.md) | What the demo must do, and the test that verifies each requirement |
| [`docs/architecture.md`](docs/architecture.md) | The pipeline, the packages, where data lives, and how to extend it |
| [`docs/data-model.md`](docs/data-model.md) | The tables, keys and rules, and how the generator behaves |
| [`AGENTS.md`](AGENTS.md) | Instructions for AI coding agents such as GitHub Copilot |

## What you need

| | Needed for | Notes |
|---|---|---|
| [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, and [Docker](https://www.docker.com/) (or another container runtime) | Everything | Or use [GitHub Codespaces](https://github.com/features/codespaces), which needs neither. Running without a container is covered in [step 1](#step-1-open-the-repository). |
| An **Azure AI Foundry** chat model deployment: its endpoint, API key and deployment name | Writing review text (step 4), and the comparison with Jev | Any chat model works, for example `gpt-4.1-mini`. You don't need an Azure login of your own, only these three values. |
| A **TypeSafe AI** API key | Classifying (step 5) | From [typesafe.ai](https://typesafe.ai). |

You can do steps 1 and 2, and read all the code and docs, without any Azure or TypeSafe account. The tests never call a live service.

## Step 1: Open the repository

1. Get the code, either way:
   - **Clone it:** `git clone https://github.com/Smartitect/retail-review-knowledge-mining.git`
   - **Or download it:** on GitHub choose **Code** → **Download ZIP**, then unzip it. You get a folder called `retail-review-knowledge-mining-main`. Git is not needed; everything the demo requires is in the zip, and the data, caches and `.env` are generated or created later.
2. Start Docker, then open the folder in VS Code (**File** → **Open Folder**). Open the folder that holds `README.md` and `pyproject.toml`, not its parent.
3. When prompted, choose **Reopen in Container**. If you are not prompted, run **Dev Containers: Reopen in Container** from the command palette.
4. Wait for the container to build. The first build takes a few minutes and needs internet access. `.devcontainer/postCreateCommand.ps1` installs [uv](https://docs.astral.sh/uv/), installs Python 3.12 and every dependency into `.venv`, and sets up the terminal to use it.

The container provides PowerShell (the default terminal), the GitHub CLI, the Azure CLI and uv. Every command in this README also works in bash. The folder name doesn't matter.

<details>
<summary>Without a dev container</summary>

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run this from the repository root:

```powershell
uv sync --python 3.12
```

uv downloads Python 3.12 if you don't have it. Run every command through `uv run`, as below.
</details>

## Step 2: Check it works

```powershell
uv run pytest
```

All tests should pass in about 10 seconds. They stub Azure AI Foundry and Jev, so they need no credentials.

## Step 3: Connect to Azure AI Foundry and Jev

Both services are reached with keys you put in `.env`. No Azure sign-in is needed.

### 3a. Find the Foundry settings

In the [Azure AI Foundry portal](https://ai.azure.com), open your resource and note three things:

- its **endpoint**, the resource root: `https://<resource>.services.ai.azure.com/`. A longer URL the portal shows for the same resource, such as `…/openai/v1/responses`, also works: only the scheme and host are used;
- its **API key**;
- the **name of your chat deployment**.

If someone else manages the Foundry resource, ask them for these three values.

### 3b. Create `.env`

```powershell
Copy-Item .env.example .env      # bash: cp .env.example .env
```

Then fill it in:

| Variable | Value |
|---|---|
| `TYPESAFE_API_KEY` | Your TypeSafe AI API key |
| `KG_REFLECTION_MODEL_ENDPOINT` | The endpoint from step 3a |
| `KG_REFLECTION_MODEL_KEY` | The API key from step 3a |
| `KG_REFLECTION_MODEL_DEPLOYMENT` | The deployment name from step 3a |
| `KG_REFLECTION_MODEL_API_VERSION` | `v1`, already set. Change it only to pin a dated Azure OpenAI version for review text; the DSPy classifier needs `v1`. |

`.env` is gitignored, so it is never committed. Treat it as a secret: it holds both keys.

### 3c. Check the connection

```powershell
uv run python -c "from dotenv import load_dotenv; load_dotenv(); from review_writer import model_settings; print(model_settings())"
```

This prints the endpoint, API version and deployment it found. The key is never printed. If it fails, the error names what is missing; see [Troubleshooting](#troubleshooting).

## Step 4: Generate the dataset

The dataset grows in batches, so you choose how much compute, storage and spend to take on. Start a dataset, then see what a batch of 1,000 customers will cost before paying for it:

```powershell
uv run generate-data init --seed 42 --as-of 2026-06-30
uv run generate-data add-customers --count 1000 --dry-run --foundry-input-price 0.15 --foundry-output-price 0.60
```

`--dry-run` writes nothing and calls nothing. The prices are your deployment's USD per million input and output tokens; leave them out and you get token counts only. When you're happy with the estimate, generate it for real:

```powershell
uv run generate-data add-customers --count 1000     # about 8 minutes: Foundry writes ~1,500 reviews
uv run generate-data status
```

The data is written to `data/generated/` (gitignored). Other commands, when you want them:

```powershell
uv run generate-data add-customers --total 5000     # idempotent: tops up to 5,000 customers
uv run generate-data advance --to 2026-12-31        # existing customers keep buying and reviewing
uv run generate-data fill-texts                     # retry any review text that failed
uv run generate-data --help
```

How it behaves:

- **Deterministic.** The same seed gives the same data, and two batches of 1,000 customers equal one batch of 2,000. Advancing time gives exactly what generating straight to the later date would have.
- **Idempotent.** `--total` and `advance --to` do nothing when the dataset is already there.
- **Safe to interrupt.** A batch only counts once `manifest.json` lists it, so an interrupted batch is invisible and is replaced on the next run.
- **Cached.** Review text is cached by prompt and model in `data/generated/review_text_cache.parquet`, so a rerun reuses it. A failed text is reported and retried, never cached.

### How far to scale

Measured on the defaults (1,000 and 10,000 customers; 100,000 extrapolated). Jev uses about 2,750 input tokens and 0.3 seconds per sentence, at $0.042 per million tokens and a concurrency of 8. Foundry times assume about 2.5 seconds per review at a concurrency of 8 (1,000 customers took 8.3 minutes on `gpt-5.6-terra`); Foundry's cost depends on your deployment's prices.

| Customers | Orders | Reviews | Sentences | Generate | Storage | Foundry tokens (in / out) | Foundry time | Jev cost | Jev time |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | ~8,400 | ~1,500 | ~6,800 | ~2 s | ~0.4 MB | 0.4M / 0.2M | ~8 min | ~$0.79 | ~4 min |
| 10,000 | ~76,000 | ~13,700 | ~61,000 | ~15 s | ~3 MB | 3.6M / 1.7M | ~70 min | ~$7 | ~40 min |
| 100,000 | ~760,000 | ~137,000 | ~610,000 | ~2.5 min | ~30 MB | 36M / 17M | ~12 h | ~$71 | ~8.5 h |

At 100,000 customers, Jev's published limit of 1,200 requests a minute sets the pace, not concurrency. Grow in batches, and classify each batch before adding the next.

## Step 5: Classify the reviews with Jev

1. Open [`notebooks/01_classify_reviews_with_jev.ipynb`](notebooks/01_classify_reviews_with_jev.ipynb).
2. Choose **Select Kernel** → **Python Environments** → `.venv`.
3. Choose **Run All**.

The notebook:

1. Tops the dataset up to `CUSTOMERS` (1,000 by default, with seed 42). If you already generated 1,000 customers in step 4, it reuses them. If you used a different seed, change `SEED` in the notebook to match.
2. Loads one row per review, with what was known about the customer at the moment they wrote it.
3. Splits each review into sentences.
4. Asks Jev 26 questions about every sentence, and writes `data/output/sentences_classified.parquet`.
5. Analyses the answers: frustration against stars, problems by product, language, escalations, suggestions, competitors, and the low-confidence sentences to send to a person.

For 1,000 customers, classification takes about 4 minutes and costs about $0.79. Answers are cached in `data/output/jev_sentence_answers.parquet` (gitignored), so running it again costs nothing. The cache is keyed on the question-set version, so changing a question in `src/jev_classifier/questions.py` means bumping `QUESTION_SET_VERSION`.

To see exactly what is sent to and received from Jev, call `jev_classifier.transcript.log_to_stdout()` before classifying. Each exchange prints as one JSON object, with the request and response bodies as they crossed the wire, the request ID and the latency. The API key is never printed. `transcript.silence()` turns it off again. The notebook's "Watch the wire" section does this for two sentences.

## Step 6: Explore the dashboard

```powershell
uv run streamlit run app/streamlit_app.py
```

VS Code offers to open the forwarded port; otherwise browse to <http://localhost:8501>. The dashboard reads `data/output/sentences_classified.parquet`, so run the notebook first.

- **Sentiment flow:** a Sankey from review sentiment through product category and product to each review's *primary issue* (the problem in its most frustrated sentence). Every review is exactly one path, and widths are review counts. Sentiment can come from the text (Jev) or from the star rating, so you can see where they disagree. A drill-down lists the reviews behind any issue.
- **Customers at risk:** one mark per customer at their most recent review, plotting tenure against lifetime revenue as at that review, shaped and coloured by risk tier. Box or lasso select to list customers. Set the risk thresholds in the sidebar.

Filter by product category, product and country across both views. The colours are endjin's, assigned by role and checked for colour-blind separation (`src/review_charts/palette.py`). They are validated for the light theme, which `.streamlit/config.toml` pins.

## Step 7 (optional): Compare Jev with a foundation model

[`notebooks/02_compare_jev_and_foundry.ipynb`](notebooks/02_compare_jev_and_foundry.ipynb) asks Jev's 26 questions of a foundation model too: the Azure AI Foundry deployment that writes the reviews, through a zero-shot [DSPy](https://dspy.ai) program built from Jev's own question set. It runs both classifiers live on the same sample of whole reviews, at the same concurrency, and compares them on:

- **speed:** latency per sentence and throughput;
- **cost:** tokens and dollars per sentence, and per 1,000 customers;
- **agreement:** per question, how often the two give the same answer, with examples where they differ;
- **accuracy:** both scored against what each review was written to say (`review_truth`).

Run notebook 01 first. Set `FOUNDRY_PRICES` to your deployment's prices to cost it, and `REASONING_EFFORT` to trade the foundation model's accuracy for speed. The notebook estimates the cost before calling anything. A 300-sentence target samples about 370 sentences: about $0.05 of Jev, and roughly 1.9 million Foundry input tokens plus the model's reasoning. Each run is saved to `data/output/comparison/`, and the analysis reuses it unless you set `RERUN = True`.

The foundation model reports its confidences rather than measuring them, so they fill the same columns as Jev's but are not calibrated.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Azure AI Foundry is not configured: set KG_REFLECTION_MODEL_…` | `.env` is missing, or the variables it names are empty. See [step 3b](#3b-create-env). |
| `401` or `Access denied due to invalid subscription key` from Foundry | `KG_REFLECTION_MODEL_KEY` is wrong, or belongs to a different resource from `KG_REFLECTION_MODEL_ENDPOINT`. |
| `404` or `DeploymentNotFound` from Foundry | `KG_REFLECTION_MODEL_DEPLOYMENT` doesn't match a deployment on that resource. Use the deployment name, not the model name. |
| `… already holds a dataset with different settings; use another directory` | The notebook's `SEED` differs from the seed `generate-data init` used. Make them match, or pass `--dir` to use another directory. |
| `N review(s) have no text yet and are left out.` | Some Foundry calls failed. Run `uv run generate-data fill-texts`. |
| `the DSPy classifier needs Foundry's v1 API` | Set `KG_REFLECTION_MODEL_API_VERSION=v1` in `.env`. DSPy's route for dated Azure OpenAI versions does not work alongside `openai` 3.x. |
| Sentences classified with errors | Jev calls failed (for example, rate limits). They are not cached: rerun the classification cell. |
| `No classified reviews at data/output/sentences_classified.parquet` in the dashboard | Run the notebook (step 5) first. |
| `python --version` reports a system Python | Open a new terminal, or run commands through `uv run`. |

## Code map

| Path | Responsibility |
|---|---|
| `src/retail_model` | Table schemas (pandera), cross-table integrity, review propensity, dataset storage |
| `src/retail_generator` | Deterministic customers, orders and reviews; every setting in `GeneratorConfig`; the `generate-data` CLI |
| `src/review_writer` | Review text from Azure AI Foundry, with settings from `.env`, cached by prompt |
| `src/customer_features` | Point-in-time features: only data from before each review, never after |
| `src/review_wrangler` | Load reviews with their text, product and features; split them into sentences |
| `src/jev_classifier` | The question set, and async classification with a Parquet cache |
| `src/dspy_classifier` | The same questions asked of a foundation model on Azure AI Foundry through DSPy, and the comparison with Jev |
| `src/review_insights` | Review and customer views, Sankey flows, risk tiers, escalation and review queues |
| `src/review_charts` | Plotly figures and colour roles for the dashboard |
| `app/streamlit_app.py` | The dashboard: layout and state only |
| `notebooks/` | `01`: the end-to-end walkthrough. `02`: Jev against a foundation model. They call `src/` and hold no logic of their own. |
| `reference_data/products.csv` | The 17-product catalogue |
| `tests/unit/` | Unit tests. They never call a live service. |

See [`docs/architecture.md`](docs/architecture.md) for how the packages depend on each other and how to extend them.

## Development

```powershell
uv add <package>            # add a runtime dependency
uv add --dev <package>      # add a development tool
uv sync                     # after a pull that changes pyproject.toml or uv.lock
uv run pytest               # unit tests
uv run ruff check           # lint
```

- **Use uv, never pip.** `pyproject.toml` is the only place dependencies are declared, and `uv.lock` is committed. `uv run` syncs the environment before it runs, so there's no need to activate the virtualenv.
- **Don't run `ruff format`.** The code is hand-formatted; `ruff check` is the gate.
- **CI** runs `ruff check` and `pytest` on every pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Pull requests** branch from `main` and describe why, what changed and how it was tested.

### Working with AI coding agents

[`AGENTS.md`](AGENTS.md) holds the instructions for coding agents: the commands, the invariants the tests protect, where code belongs, and the rules on cost and secrets.

- **GitHub Copilot** reads `AGENTS.md` and [`.github/copilot-instructions.md`](.github/copilot-instructions.md). The dev container installs Copilot and turns on `chat.useAgentsMdFile`, so agent mode in VS Code picks `AGENTS.md` up. [`.github/workflows/copilot-setup-steps.yml`](.github/workflows/copilot-setup-steps.yml) prepares the environment for Copilot's cloud agent, so it can run the tests from its first step.
- **Claude Code** reads [`CLAUDE.md`](CLAUDE.md), which imports `AGENTS.md`.

### Smoke test after rebuilding the container

```powershell
uv --version
python --version   # must report 3.12 from .venv, not a system interpreter
gh --version
az --version
pytest --version
```

Rebuild *without cache* now and then: a container definition that only works incrementally is broken for the next person who clones it.

## License

[MIT](LICENSE)
