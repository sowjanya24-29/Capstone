"""Zepto-style catalog pipeline: scrape, clean, convert, store, query.

Scrapes books.toscrape.com (a public scraping-practice site), cleans the
rows, converts GBP to INR at the project's fixed rate, and loads a
normalized SQLite database. Then runs the required SQL and pandas checks.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
GBP_TO_INR = 105.50  # project-defined constant, not a market rate
MIN_BOOKS = 60
MIN_CATEGORIES = 3
HEADERS = {
    "User-Agent": "capstone-data-pipeline/1.0 (educational scraping practice)"
}
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
OUTPUT_PATH = ROOT / "query_outputs.txt"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    # Use the raw bytes so BeautifulSoup honors the page's charset.
    # response.text was mis-decoding apostrophes in some titles.
    return BeautifulSoup(response.content, "html.parser")


def list_categories() -> list[tuple[str, str]]:
    soup = fetch(BASE_URL)
    categories = []
    for link in soup.select("div.side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        href = urljoin(BASE_URL, link["href"])
        categories.append((name, href))
    if len(categories) < MIN_CATEGORIES:
        raise RuntimeError(f"Expected at least {MIN_CATEGORIES} categories, found {len(categories)}")
    return categories


def scrape_category(category: str, start_url: str) -> list[dict]:
    books = []
    page_url = start_url
    while page_url:
        soup = fetch(page_url)
        for pod in soup.select("article.product_pod"):
            title = pod.select_one("h3 a")["title"].strip()
            price_text = pod.select_one("p.price_color").get_text(strip=True)
            rating_classes = pod.select_one("p.star-rating")["class"]
            rating_text = next((c for c in rating_classes if c != "star-rating"), "")
            availability = pod.select_one("p.instock.availability").get_text(" ", strip=True)
            books.append(
                {
                    "title": title,
                    "price_text": price_text,
                    "rating_text": rating_text,
                    "availability_text": availability,
                    "category": category,
                }
            )
        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None
        time.sleep(0.2)
    return books


def scrape_books() -> pd.DataFrame:
    """Scrape whole categories until we have >= 60 books across >= 3 categories."""
    rows: list[dict] = []
    used = 0
    for name, url in list_categories():
        if used >= MIN_CATEGORIES and len(rows) >= MIN_BOOKS:
            break
        print(f"Scraping category: {name}")
        rows.extend(scrape_category(name, url))
        used += 1
    frame = pd.DataFrame(rows)
    print(f"Scraped {len(frame)} books across {frame['category'].nunique()} categories")
    return frame


def _parse_price(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _parse_rating(value: str) -> float | None:
    return float(RATING_MAP[value]) if value in RATING_MAP else None


def _parse_stock(value: str) -> bool | None:
    text = str(value).lower()
    if "in stock" in text:
        return True
    if "out of stock" in text or "unavailable" in text:
        return False
    return None


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the scraped fields. Unparseable numerics are median-imputed.

    Median imputation keeps a row whose only problem is a stray price or
    rating token, instead of dropping an otherwise valid book. Availability
    that cannot be parsed is dropped, because a guessed boolean would invent
    stock status the page did not state.
    """
    frame = raw.copy()
    frame["price_gbp"] = frame["price_text"].map(_parse_price)
    frame["rating"] = frame["rating_text"].map(_parse_rating)
    frame["in_stock"] = frame["availability_text"].map(_parse_stock)

    before = len(frame)
    frame = frame.dropna(subset=["in_stock", "title", "category"]).copy()
    dropped = before - len(frame)

    for column in ("price_gbp", "rating"):
        missing = int(frame[column].isna().sum())
        if missing:
            median = float(frame[column].median())
            frame[column] = frame[column].fillna(median)
            print(f"Median-imputed {missing} values in {column} with {median}")

    frame["price_gbp"] = frame["price_gbp"].astype(float)
    frame["rating"] = frame["rating"].round().astype(int)
    frame["in_stock"] = frame["in_stock"].astype(bool)
    frame["price_inr"] = (frame["price_gbp"] * GBP_TO_INR).round(2)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable availability or missing title/category")
    return frame[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].reset_index(drop=True)


def load_database(frame: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            );
            """
        )
        categories = (
            frame[["category"]]
            .drop_duplicates()
            .sort_values("category")
            .reset_index(drop=True)
        )
        categories.index = categories.index + 1
        categories.index.name = "category_id"
        categories = categories.reset_index().rename(columns={"category": "category_name"})
        categories.to_sql("categories", conn, if_exists="append", index=False)

        name_to_id = dict(zip(categories["category_name"], categories["category_id"]))
        books = frame.copy()
        books["category_id"] = books["category"].map(name_to_id)
        books["in_stock"] = books["in_stock"].astype(int)
        books = books.drop(columns=["category"])
        books.index = books.index + 1
        books.index.name = "book_id"
        books = books.reset_index()
        books.to_sql("books", conn, if_exists="append", index=False)
        conn.execute(
            "CREATE INDEX idx_books_category_id ON books(category_id)"
        )
    print(f"Loaded {len(frame)} books into {DB_PATH.name}")


QUERIES: list[tuple[str, str]] = [
    (
        "in_stock_high_rated",
        """
        SELECT title, rating, price_gbp, price_inr
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, title
        """,
    ),
    (
        "ten_highest_inr_prices",
        """
        SELECT title, price_inr, rating
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
        """,
    ),
    (
        "distinct_categories",
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
        """,
    ),
    (
        "mid_price_or_top_rating",
        """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
           OR rating IN (4, 5)
        ORDER BY rating DESC, price_gbp
        """,
    ),
    (
        "top_rated_per_category",
        """
        SELECT category_name, title, rating, price_inr
        FROM (
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_inr,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_id
                    ORDER BY b.rating DESC, b.price_inr DESC, b.title
                ) AS rn
            FROM books AS b
            JOIN categories AS c ON b.category_id = c.category_id
        )
        WHERE rn <= 10
        ORDER BY category_name, rating DESC, price_inr DESC, title
        """,
    ),
]


def run_queries() -> str:
    sections = [
        "SQL query log",
        f"Fixed conversion used for price_inr: 1 GBP = {GBP_TO_INR:.2f} INR",
        "",
    ]
    frames: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for name, sql in QUERIES:
            result = pd.read_sql(sql, conn)
            frames[name] = result
            sections.append(f"=== {name} ===")
            sections.append(sql.strip())
            sections.append(result.to_string(index=False))
            sections.append(f"rows: {len(result)}")
            sections.append("")

        join_sql = dict(QUERIES)["top_rated_per_category"]
        read_sql_join = pd.read_sql(join_sql, conn)
        where_sql = pd.read_sql(dict(QUERIES)["in_stock_high_rated"], conn)
        books = pd.read_sql("SELECT * FROM books", conn)
        categories = pd.read_sql("SELECT * FROM categories", conn)

    merged = books.merge(categories, on="category_id", how="inner")
    merged["rn"] = (
        merged.sort_values(
            ["rating", "price_inr", "title"], ascending=[False, False, True]
        )
        .groupby("category_id")
        .cumcount()
        + 1
    )
    pandas_join = (
        merged.loc[merged["rn"] <= 10, ["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "rating", "price_inr", "title"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    sql_join = read_sql_join.reset_index(drop=True)
    match = sql_join.equals(pandas_join)

    sections.append("=== pandas read_sql: in_stock_high_rated ===")
    sections.append(where_sql.to_string(index=False))
    sections.append("")
    sections.append("=== pandas read_sql: top_rated_per_category ===")
    sections.append(sql_join.to_string(index=False))
    sections.append("")
    sections.append("=== pandas merge: top_rated_per_category ===")
    sections.append(pandas_join.to_string(index=False))
    sections.append("")
    sections.append(f"read_sql JOIN equals pd.merge JOIN: {match}")
    if not match:
        sections.append("SQL rows missing from pandas:")
        sections.append(
            sql_join.merge(pandas_join, how="left", indicator=True)
            .query("_merge != 'both'")
            .to_string(index=False)
        )
    text = "\n".join(sections) + "\n"
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.name}; join outputs match: {match}")
    if not match:
        raise SystemExit("pandas merge did not match the SQL join")
    return text


def main() -> None:
    raw = scrape_books()
    cleaned = clean(raw)
    if len(cleaned) < MIN_BOOKS or cleaned["category"].nunique() < MIN_CATEGORIES:
        raise SystemExit(
            f"Need >= {MIN_BOOKS} books and >= {MIN_CATEGORIES} categories; "
            f"got {len(cleaned)} books and {cleaned['category'].nunique()} categories"
        )
    print(cleaned.dtypes.to_string())
    print(
        f"price_inr check: first row {cleaned.iloc[0]['price_gbp']} GBP -> "
        f"{cleaned.iloc[0]['price_inr']} INR"
    )
    load_database(cleaned)
    run_queries()


if __name__ == "__main__":
    main()
