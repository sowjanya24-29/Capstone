"""Local FastAPI wrapper around the Zepto policy graph."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from assistant import AskRequest, AskResponse, ask, ingest

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ingest()
    yield


app = FastAPI(title="Zepto support assistant", lifespan=lifespan)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    return ask(body.query)
