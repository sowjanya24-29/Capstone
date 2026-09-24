# Zepto data and AI platform

Capstone for the Certificate Program in Artificial Intelligence and Machine Learning. One repository, three modules:
## Project Structure

Capstone/
├── data_pipeline/
├── analytics/
├── support_assistant/
├── README.md
└── .gitignore

| Folder | Role | Marks |
| --- | --- | --- |
| `data_pipeline` | Scrape, clean, convert, and query catalog data | 25 |
| `analytics` | Profile the Titanic dataset and model it | 50 |
| `support_assistant` | Grounded policy assistant | 25 |

Each module has its own `requirements.txt`. Use a virtual environment so these installs do not change other Python projects on the machine.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install and run a module from its folder.

## Setup and run

Data pipeline:

```powershell
cd data_pipeline
python -m pip install -r requirements.txt
python pipeline.py
```

Analytics (run the EDA script before modeling; modeling only reads `titanic.csv`):

```powershell
cd analytics
python -m pip install -r requirements.txt
python 01_eda.py
python 02_modeling.py
```

The same flow is also available as `01_eda.ipynb` then `02_modeling.ipynb`.

Support assistant (graded path leaves `MOCK_LLM` unset, which means mock mode):

```powershell
cd support_assistant
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 7860
```

Then `POST /ask` with `{"query": "..."}`. Container form of the same app:

```powershell
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

## Design decisions

**Data pipeline.** Live pages from books.toscrape.com are cleaned into typed columns and priced in INR at the fixed rate 1 GBP = 105.50 INR. The relational store is two SQLite tables, `categories` and `books`, joined on `category_id`. SQL and a pandas merge are both used for the same per-category ranking so the two paths can be checked against each other. Details and the query log are in `data_pipeline/README.md` and `data_pipeline/query_outputs.txt`.

**Analytics.** The Titanic file is loaded once and saved as `analytics/titanic.csv`. Exploratory cleaning follows the percentage rules (drop rows under 5% missing, impute between 5% and 30%, drop `deck` because it is 77% empty). Modeling does not repeat that row-dropping. Its imputer, encoder, and scaler sit inside a scikit-learn pipeline that is fit on the stratified training split only. Written interpretations, the comparison tables, and the deploy recommendation are in `analytics/README.md`.

**Support assistant.** Eight Zepto policy files are chunked one document at a time, embedded locally with `all-MiniLM-L6-v2`, and stored in ChromaDB. A LangGraph flow classifies the query, then either retrieves the top three chunks or refuses. With `MOCK_LLM` left at the default, no LLM is called. Answers are a fixed template plus a Pydantic `answer` / `sources` / `confidence` payload, served by FastAPI.. The architecture write-up and example responses are in `support_assistant/README.md`.
