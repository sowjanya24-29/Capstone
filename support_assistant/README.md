# Support assistant

A local Zepto policy assistant. The graded path does not call an LLM. Leave `MOCK_LLM` unset (the code treats that as mock mode, same as `MOCK_LLM=1`).

## Setup

```powershell
cd support_assistant
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 7860
```

Open http://127.0.0.1:7860/ for the small chat page, or `POST /ask` with `{"query": "<text>"}`. API docs stay at `/docs`.

Docker, from this folder:

```powershell
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

The image downloads `all-MiniLM-L6-v2` during `docker build`. The container listens on `0.0.0.0:7860`. Pushing the image to Hugging Face Spaces is optional and not required.

ChromaDB is created at startup in `chroma/` (gitignored) from `docs/doc_01.txt` … `docs/doc_08.txt`. One chunk per file. Embeddings are `sentence-transformers` `all-MiniLM-L6-v2`, stored with cosine distance.

## Example calls

Recorded against `http://127.0.0.1:7860/ask` with `MOCK_LLM` left unset.

Policy question (keyword `delivery` routes to retrieval):

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}
```

The top chunk is `doc_01`, the delivery policy, and the answer is the canned template plus the first 200 characters of that chunk. No LLM call is made.

Unrelated question (no policy keyword, routes to the direct answer):

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Prompt template

The optional `MOCK_LLM=0` path fills this template, which lives as text in `PROMPT_TEMPLATE` inside `assistant.py`:

- Role: Zepto's policy support assistant, answering only from the excerpts it is given.
- Context: the retrieved chunks.
- Task: answer the customer query using only that context.
- Format: one JSON object with `answer`, `sources`, and `confidence`.
- Length: under 80 words.
- Negative constraint: do not answer using information not present in the provided context.
- Few-shot example: a packed-order cancellation question answered from `doc_05`.

Mock mode does not send this prompt anywhere. If `MOCK_LLM=0`, a Groq chat completion is called with `GROQ_API_KEY` (never committed). If that output fails the Pydantic schema, the call is retried up to two more times and then a marked error response is returned.

## Architecture

Ingestion. `ingest()` reads the eight files in `docs/`, treats each file as one chunk, and stores `doc_01` … `doc_08` in the Chroma collection `zepto_policies`. This runs on FastAPI startup and again at the start of `ask()`, and it skips the insert when the collection already has all eight documents.

Embedding. `get_collection()` builds a `SentenceTransformerEmbeddingFunction` for `all-MiniLM-L6-v2` and attaches it to that collection. The same function embeds queries at retrieval time. This stage does not look at `MOCK_LLM`.

Retrieval. The LangGraph node `retrieve_and_answer` embeds the query and asks Chroma for the top 3 chunks by cosine similarity. It runs only after `classify_intent` returns `policy_question`. Retrieval is real in both mock mode and the optional real-LLM mode.

Generation. This is the stage that branches on `MOCK_LLM`.

- `classify_intent` runs first. In mock mode it is a keyword check (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`) and makes no LLM call. A hit is `policy_question`; anything else is `general_question`. A conditional edge then goes to `retrieve_and_answer` or `direct_answer`.
- `retrieve_and_answer` in mock mode returns `Based on the retrieved context: ` plus a 200-character excerpt of the top chunk, `sources` set to the retrieved ids, and `confidence` 1.0. With `MOCK_LLM=0` it fills `PROMPT_TEMPLATE` and parses the model output into the same schema.
- `direct_answer` in mock mode returns the fixed sentence `I can only answer questions about Zepto policies right now.` with an empty `sources` list. With `MOCK_LLM=0` it calls the model with no retrieved policy text.

`POST /ask` in `main.py` validates the request and returns that Pydantic object. The graph state is a `TypedDict` with the query, the intent, and the answer fields.
