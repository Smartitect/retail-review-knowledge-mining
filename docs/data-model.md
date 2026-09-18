# Retail data model

This dataset replaces the flat review file with a small, realistic retail model. Customers buy products over time, and review a biased subset of what they bought. Tracked in #1; this note covers #2.

```mermaid
erDiagram
    customers ||--o{ orders : places
    orders ||--|{ order_lines : contains
    products ||--o{ order_lines : "sold as"
    order_lines ||--o| reviews : "reviewed in"
    customers ||--o{ reviews : writes
    products ||--o{ reviews : about
    reviews ||--|| review_texts : "text of"
    reviews ||--|| review_truth : "intended as"
```

| Table | Key | What it holds |
|---|---|---|
| `customers` | `customer_id` `C0000001` | Fictional person: name, contact, address, country and locale, gender, date of birth, `signup_at` |
| `products` | `product_id` `P001` | The catalogue: SKU, name, category, description, list price, fuel type, launch date |
| `orders` | `order_id` `O000000001` | Who bought, when (`ordered_at`), `order_total` |
| `order_lines` | `order_line_id` `L0000000001` | One product on an order: quantity, unit price at the time, line total |
| `reviews` | `review_id` `R000000001` | Who reviewed which purchase, when (`reviewed_at`), and the stars. Nothing else. |
| `review_texts` | `review_id` + `prompt_hash` | The words of the review, their language, and what wrote them |
| `review_truth` | `review_id` | What the generator intended: hidden satisfaction, the aspect to focus on, the language asked for |

Schemas are pandera models in `src/retail_model/tabular_schemas.py`. Rules that span tables are in `src/retail_model/integrity.py`:

- Every foreign key resolves.
- Orders are placed on or after signup, and totals equal the sum of their lines.
- A review is by the customer who bought the line, about the product on that line.
- A review is written after the order it reviews.

## Decisions

- **String keys with a type prefix.** `C…`, `P…`, `O…`, `L…`, `R…` are self-describing in logs and joins, and cannot be mistaken for one another.
- **Reviews do not store customer facts.** Tenure and lifetime value are derived from `customers` and `orders` *as at the moment of each review*, using only data from before it (#5). That keeps them consistent and free of leakage for machine learning.
- **Timestamps, not dates.** `signup_at`, `ordered_at` and `reviewed_at` are naive UTC timestamps, so events on the same day still have an order.
- **Text in its own table.** Review text comes from a separate, slow, costly step: a model on Azure AI Foundry. It is cached by prompt, so the rest of the dataset can be regenerated without touching it.
- **Truth in its own table.** The generator knows how satisfied each reviewer really was. That is invaluable for measuring how well Jev recovers it, and fatal if it leaks into a feature. A separate table makes using it deliberate.

## Who leaves a review

People review when they are delighted or when something went wrong, rarely when an item was just fine. Each purchase gets a hidden satisfaction *s* in [0, 1], drawn around the product's own quality. The chance of a review is U-shaped over *s*:

```
p(review) = base_rate + extremity_rate × |2s − 1| ^ sharpness
```

| Parameter | Default | Effect |
|---|---|---|
| `base_rate` | 0.03 | Chance of reviewing a purchase that was exactly "fine" (*s* = 0.5) |
| `extremity_rate` | 0.45 | Extra chance at either extreme, so *s* = 0 or 1 gives 0.48 |
| `sharpness` | 2.5 | 1 is a V; higher keeps the middle flat and pushes reviews to the extremes |
| `rating_noise` | 0.35 | Standard deviation, in stars, of the noise between satisfaction and rating |

Stars follow from satisfaction: `round(1 + 4s + noise)`, clipped to 1–5. Parameters live in `ReviewPropensity` (`src/retail_model/review_propensity.py`).

The result is a review corpus that over-represents 1★ and 5★ against the purchases underneath it. That gap is realistic, and it is worth showing.

## Review text

Text is written from a **brief**: product, stars, how the customer feels, the aspect to focus on, the language and a length. See `src/review_writer`.

- **Azure AI Foundry** (`FoundryReviewWriter`) is the intended source. It uses the OpenAI-compatible `/openai/v1` API, so any chat deployment works. Configure `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY` and `AZURE_FOUNDRY_DEPLOYMENT` in `.env`; they are never committed.
- **Templates** (`TemplateReviewWriter`) reuse the 168 hand-written reviews in `data/input/product_reviews.json`, matched on product and nearest rating. This is for offline runs and tests only. It is chosen explicitly, never as a silent fallback, and its rows say `text_source = "template"`. It writes English only.
- **Caching.** Texts are cached on `review_id` + `prompt_hash`. The hash covers the model and the rendered prompt, so the same seed reuses the same text: reproducible even though the model is not. A different writer or model, or a changed brief, writes afresh. Each row records `text_source`, `model` and `generated_at`. Failed calls are reported and retried on the next run, never cached.

## Assumptions

- Currency is not modelled; prices are plain numbers.
- One review per purchase at most. A second purchase of the same product can be reviewed separately.
- Countries are the twelve from the original sample. Faker has no Costa Rican or South African English locale, so those customers use `es_MX` and `zu_ZA` names and addresses (#3).
