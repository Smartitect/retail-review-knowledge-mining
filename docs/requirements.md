# Requirements

What the demo must do, and how we know it does. Each requirement names the test that verifies it, where one exists. Tests are in `tests/unit/`, written as `file::test`.

For how the system meets these requirements, see [`architecture.md`](architecture.md). For the tables, see [`data-model.md`](data-model.md).

## Purpose

Show how to mine actionable insight from semi-structured customer feedback: free-text product reviews in several languages, tied to structured customer and order data. The demo is built around a fictional barbecue retailer, so it can be shared, rerun and scaled without any real customer data.

It has to answer, credibly:

- **Product teams:** which products have which problems, how frustrated customers are about them, and what customers suggest.
- **Customer teams:** which customers are at risk of leaving, and how much revenue they represent.
- **Data teams:** how to get typed, auditable answers from text with a model like TypeSafe AI's Jev, and how to check those answers against the truth.

## Scope

**In scope:**

- A synthetic retail dataset: customers, orders, order lines, reviews, review text and the generator's ground truth.
- Sentence-level classification of every review with Jev.
- Review- and customer-level analysis in a notebook and an interactive dashboard.

**Out of scope:**

- Real customer data, or connectors to a real retail system.
- Production hosting, authentication or multi-user access for the dashboard.
- Model training. The point-in-time features are prepared so that a model could be trained without leakage, but none is.

## Functional requirements

### Dataset generation

| ID | Requirement | Verified by |
|---|---|---|
| GEN-01 | Generate customers, orders, order lines and reviews as related tables, with string keys that carry a type prefix (`C`, `P`, `O`, `L`, `R`). | `test_customers.py::test_ids_follow_the_numbering`, `test_retail_model.py::test_customer_ids_must_have_the_prefix` |
| GEN-02 | Generate fictional customers with Faker, using the locale of each of 12 countries, with country weights, gender share and an age distribution that can be configured. | `test_customers.py::test_every_country_generates_valid_customers`, `::test_country_weights_are_respected`, `::test_signups_and_ages_are_in_range` |
| GEN-03 | Generated emails must use Faker's reserved example domains, so none can reach a real inbox. | `test_customers.py::test_emails_only_use_reserved_example_domains` |
| GEN-04 | Generate realistic buying patterns: seasonality that flips between hemispheres, holiday peaks, rare grill purchases, accessories bought with new grills, fuel that matches the grill, and restocking on a cycle. | `test_orders.py::test_orders_peak_in_each_hemispheres_summer`, `::test_seasonality_flips_between_hemispheres`, `::test_black_friday_is_busier_than_early_november`, `::test_grill_buyers_buy_the_matching_fuel_and_a_cover`, `::test_consumables_repeat_per_customer` |
| GEN-05 | Never sell a product before its launch date. | `test_orders.py::test_nothing_is_sold_before_launch` |
| GEN-06 | Reviews must be a biased subset of purchases: the chance of a review is U-shaped over hidden satisfaction, and stars rise with satisfaction and stay within 1–5. | `test_retail_model.py::test_review_propensity_is_u_shaped`, `::test_rating_rises_with_satisfaction_and_stays_in_range` |
| GEN-07 | Record the generator's intent for each review (satisfaction, aspect, language) in a separate `review_truth` table, so classifiers can be scored against it. | `data-model.md` |
| GEN-08 | Have about a third of customers in non-English-speaking countries write in their own language, including Zulu, which Jev does not offer, to test how it handles an unknown language. | Notebook, Language section |
| GEN-09 | Keep every rate, weight and date in `GeneratorConfig`, with nothing hard-coded elsewhere. | Code review |

### Growing the dataset

| ID | Requirement | Verified by |
|---|---|---|
| INC-01 | Grow the dataset in batches, by adding customers (`add-customers`) or moving time forward (`advance`). | `test_incremental.py::test_cli_end_to_end` |
| INC-02 | Batching must not change the data: two batches of customers must equal one, and advancing from t1 to t2 must equal generating straight to t2. | `test_incremental.py::test_two_batches_of_customers_equal_one`, `::test_advancing_time_equals_generating_straight_to_the_later_date`, `test_orders.py::test_generating_in_two_steps_equals_generating_once` |
| INC-03 | An advance must also produce reviews of purchases made before it. | `test_incremental.py::test_an_advance_reviews_earlier_purchases_too` |
| INC-04 | `--total` and `advance --to` must be idempotent. | `test_incremental.py::test_total_and_advance_are_idempotent` |
| INC-05 | A batch must be atomic: files from an interrupted batch are never read, and are replaced on the next run. | `test_incremental.py::test_files_from_an_interrupted_batch_are_ignored_then_replaced` |
| INC-06 | Refuse to add to a dataset with a different seed or generator config, or to start a dataset in a directory that already holds tables without a manifest. | `test_incremental.py::test_different_settings_are_refused`, `::test_init_refuses_a_directory_of_tables_without_a_manifest` |
| INC-07 | Before writing, report each batch's review count, sentence count, and the tokens, cost and time to write and classify it. `--dry-run` reports and writes nothing. | `test_incremental.py::test_estimate_scales_with_sentences`, `::test_a_dry_run_writes_nothing` |

### Data integrity

| ID | Requirement | Verified by |
|---|---|---|
| INT-01 | Validate every table against its schema on write and on read. | `test_retail_model.py::test_line_total_must_be_quantity_times_price`, `::test_customer_ids_must_have_the_prefix` |
| INT-02 | Enforce the cross-table rules before committing a batch: every foreign key resolves, orders come on or after signup, order totals equal the sum of their lines, and each review is by the buyer, about the product on the line, after the order. | `test_retail_model.py::test_a_consistent_dataset_passes`, `::test_each_broken_rule_is_named` |
| INT-03 | Report every broken rule at once, with sample keys, rather than stopping at the first. | `test_retail_model.py::test_each_broken_rule_is_named` |

### Review text

| ID | Requirement | Verified by |
|---|---|---|
| TXT-01 | Write each review's text with a chat model on Azure AI Foundry, from a brief: product, stars, feeling, aspect, language and length. | `test_review_writer.py::test_foundry_writer_sends_the_prompt_to_the_deployment` |
| TXT-02 | Read the Foundry endpoint, key, API version and deployment from Azure Key Vault, using the developer's Azure identity. `.env` holds only the vault URI and a secret prefix. | `test_review_writer.py::test_model_settings_come_from_the_prefixed_secret_bundle` |
| TXT-03 | Support both the version-less `v1` API and dated Azure OpenAI API versions. | `test_review_writer.py::test_foundry_writer_routes_by_api_version` |
| TXT-03a | Support reasoning deployments, which accept only their default temperature: send no temperature unless one is given. | Not yet tested. Verified live on `gpt-5.6-terra`. |
| TXT-04 | Refuse to run when Foundry is unconfigured, with a message naming what is missing. Never fall back to another source of text. | `test_review_writer.py::test_foundry_writer_explains_missing_configuration` |
| TXT-05 | Cache text by review and prompt hash, where the hash covers the model and the prompt. Reuse it on a rerun, and write afresh when either changes. | `test_review_writer.py::test_prompt_is_deterministic_and_hash_covers_model`, `::test_cache_reuses_text_and_regenerates_when_the_prompt_changes` |
| TXT-06 | Never cache a failed call. Retry it on the next batch, or with `fill-texts`. | `test_review_writer.py::test_failures_are_not_cached`, `test_incremental.py::test_failed_text_is_retried_by_fill_texts` |

### Point-in-time features

| ID | Requirement | Verified by |
|---|---|---|
| PIT-01 | Derive each review's customer features (tenure, lifetime revenue, order count, days since last order, previous reviews, age) only from events strictly before the review. | `test_customer_features.py::test_orders_after_the_review_do_not_count`, `::test_an_order_at_exactly_the_review_time_does_not_count`, `::test_a_second_review_sees_the_first_ones_order_but_not_later_ones` |
| PIT-02 | No future data may change any feature. | `test_customer_features.py::test_future_data_cannot_change_any_feature` |
| PIT-03 | Features must match a brute-force calculation. | `test_customer_features.py::test_features_match_a_brute_force_calculation` |

### Loading and sentences

| ID | Requirement | Verified by |
|---|---|---|
| LOAD-01 | Load one row per review, with its text, product and point-in-time features. | `test_review_wrangler.py::test_one_row_per_review_with_text_product_and_point_in_time_features` |
| LOAD-02 | Leave out reviews that have no text yet, and say how many were left out. | `test_review_wrangler.py::test_reviews_without_text_are_left_out_and_counted` |
| LOAD-03 | Split reviews into sentences, including on the full-width `。！？` used in Chinese and Japanese. | `test_review_wrangler.py::test_split_sentences`, `::test_split_sentences_on_full_width_punctuation` |

### Classification with Jev

| ID | Requirement | Verified by |
|---|---|---|
| JEV-01 | Ask Jev 26 typed questions about every sentence: frustration (Score 0–4), problem category, language, sentiment and recommendation (Choices), churn risk, safety concern, suggestion and competitor mention (Nouls), and one mention Noul per catalogue product. | `test_jev_classifier.py::test_one_mention_question_per_product` |
| JEV-02 | Send the sentence and its whole review as context, but never the star rating or the customer's demographics. | `test_jev_classifier.py::test_state_leaves_out_rating_and_demographics` |
| JEV-03 | Flatten each sentence's answers into one row, and keep every probability distribution as JSON for audit and for re-thresholding. | `test_jev_classifier.py::test_answers_become_one_flat_row`, `::test_mentions_below_threshold_are_left_out` |
| JEV-04 | Ask about each distinct sentence once, and cache answers under the question-set version, so a rerun only pays for new sentences. | `test_jev_classifier.py::test_duplicate_sentences_are_asked_once`, `::test_cached_sentences_are_not_asked_again` |
| JEV-05 | Record a failed call with its error and nulls, never fill it in, and never cache it. | `test_jev_classifier.py::test_a_failure_is_recorded_not_filled_in_and_not_cached` |
| JEV-06 | Optionally print every exchange with Jev exactly as it crossed the wire. Off by default, and never printing the API key. | `test_transcript.py` (all six tests) |

### Insight and dashboard

| ID | Requirement | Verified by |
|---|---|---|
| INS-01 | Roll sentences up to one row per review: peak and mean frustration, problems, products mentioned and flags, with the flag threshold passed as an argument. | `test_customer_view.py::test_one_row_per_review_with_derived_columns` |
| INS-02 | Derive review sentiment from the text (Jev) and from the stars, so the two can be compared. | `test_customer_view.py::test_one_row_per_review_with_derived_columns` |
| INS-03 | Give each review one primary issue (the problem in its most frustrated sentence), so the Sankey conserves reviews at every hop. | `test_customer_view.py::test_sankey_conserves_reviews_at_every_hop` |
| INS-04 | Assign risk tiers (At risk, Frustrated, Satisfied) from churn signals and peak frustration, with thresholds the user can change. Judge each customer on their most recent review. | `test_customer_view.py::test_risk_tiers`, `::test_a_customer_is_their_most_recent_review` |
| INS-05 | Provide work queues: safety and churn escalations, suggestions, competitor mentions, cross-product mentions, and low-confidence sentences to route to a person. | Notebook, section 5 |
| INS-06 | Show a dashboard with a sentiment Sankey (sentiment → category → product → primary issue) and a customers-at-risk scatter, filterable by category, product and country, with drill-down to the reviews behind each. | Manual: `uv run streamlit run app/streamlit_app.py` |

## Non-functional requirements

| ID | Requirement | How it is met |
|---|---|---|
| NFR-01 **Reproducibility** | The same seed gives the same structured data, and the same cached text, on any machine. | Per-entity random streams; text cached by prompt hash (INC-02, TXT-05) |
| NFR-02 **Cost transparency** | No paid call without an estimate first; reruns cost nothing for work already done. | `estimate.py`, `--dry-run`, both caches (INC-07, TXT-05, JEV-04) |
| NFR-03 **Scale** | Run on a laptop from 1,000 to 100,000 customers; classify in batches within Jev's limit of 1,200 requests a minute. | Polars in memory; concurrency 8; sizing table in the README |
| NFR-04 **No leakage** | Features never see the future; ground truth is never a feature. | PIT-01 to PIT-03, GEN-07 |
| NFR-05 **Security** | No secret in git or in logs. Foundry secrets live in Key Vault, accessed with the developer's identity. | `.env` gitignored; `ModelSettings.__repr__` hides the key; transcript omits headers (TXT-02, JEV-06) |
| NFR-06 **Testability** | Every unit test runs offline, with no credentials and no live service. | Stubs for Foundry and Jev; enforced by CI |
| NFR-07 **Accessibility** | Dashboard colours are distinguishable with colour-vision deficiency, and colour never carries meaning alone. | `review_charts/palette.py`: validated palette, plus a marker shape per risk tier |
| NFR-08 **Reproducible environment** | A fresh clone builds and passes its tests with no manual setup beyond credentials. | Dev container, `uv.lock`, CI |

## Constraints

- **Python 3.12**, managed by **uv**. `pyproject.toml` is the only place dependencies are declared, and `uv.lock` is committed.
- **Azure AI Foundry** for review text, with its settings in **Azure Key Vault**. The developer needs an Azure identity that can read the vault's secrets.
- **TypeSafe AI Jev** for classification, with a `TYPESAFE_API_KEY`.
- **Timestamps are naive UTC** throughout the data model.

## Assumptions

- Currency is not modelled; prices are plain numbers.
- A purchase is reviewed at most once. A second purchase of the same product can be reviewed separately.
- The 12 countries and 17 products come from the demo's original sample. Faker has no Costa Rican or South African English locale, so those customers use the `es_MX` and `zu_ZA` locales.
- Generated review text is clean enough for a regex sentence splitter.

## Possible next steps

These are not requirements yet. They are where the demo could go next:

- Score Jev against `review_truth` systematically (aspect against problem category, language against language) and report precision and recall.
- Replace the regex sentence splitter with a proper segmenter, ready for real reviews.
- Run the pipeline on a schedule, writing to a lakehouse rather than local Parquet.
- Use the point-in-time features to train and evaluate a churn model.
