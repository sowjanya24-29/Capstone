# Data pipeline

Scrape practice-catalog books, clean the rows, convert GBP to INR at the fixed project rate, and load a two-table SQLite database.

## Setup

```powershell
cd data_pipeline
python -m pip install -r requirements.txt
python pipeline.py
```

The script writes `books.db` and `query_outputs.txt` next to itself. Re-running replaces the database.

## Design decisions

- Source: `http://books.toscrape.com/`, scraped with `requests` and BeautifulSoup. No API key.
- Scope: the Travel, Mystery, and Historical Fiction categories, including their paginated pages. That yields 69 books across 3 categories.
- Captured fields: title, listed GBP price, star-rating text, availability text, and category.
- `price_gbp` is the listed price as a float. `rating` maps One…Five to 1…5. `in_stock` is true when the availability text says "In stock".
- `price_inr` uses the required fixed rate **1 GBP = 105.50 INR**. This is a project constant, not a market rate, and it does not depend on a date or an API.
- If a numeric field fails to parse, the pipeline median-imputes that column so one bad token does not drop an otherwise usable book. If availability, title, or category cannot be parsed, the row is dropped, because a guessed stock flag would invent a fact the page did not state. On the current scrape every row parsed, so neither path changed the row count.
- Schema: `categories(category_id, category_name)` and `books(..., category_id)` with a foreign key. Titles are not unique on the site, so the key is a surrogate `book_id`.
- Five SQL statements in `pipeline.py` cover `SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and a `JOIN`. The per-category top-rated query is also rebuilt with `pd.merge`. `query_outputs.txt` records both outputs and that they match.
