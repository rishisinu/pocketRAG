from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.core.ingestion import process_and_store_data
from app.core.LLM_query import answer_query
from app.core.models import IngestResult, QueryRequest, QueryResponse
from app.core.query_crossencoder import query_handler

import asyncio
import os
import subprocess
import time
import urllib.request
import urllib.error
from contextlib import asynccontextmanager

SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN")
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH")
l_port = int(os.environ.get("LLAMA_PORT", 8080))
n_ctx = int(os.environ.get("LLAMA_CTX", 4096))
llama_health = f"http://localhost:{l_port}/health"
llama_process: subprocess.Popen[bytes] | None = None


def starting_llama_server(timeout: float = 120, interval: float = 0.5):
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        # if the process fell over there is no point sitting here for the full timeout
        if llama_process is not None and llama_process.poll() is not None:
            raise RuntimeError(f"llama cpp exited with code {llama_process.returncode}")
        try:
            urllib.request.urlopen(llama_health, timeout=1)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(interval)
    raise RuntimeError("llama cpp didnt boot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_process

    # If the binary/model arent configured we just assume the user is running
    # their own llama-server already and let the client talk to it.
    if SERVER_BIN and MODEL_PATH:
        llama_process = subprocess.Popen([
            SERVER_BIN,
            "-m", MODEL_PATH,
            "--port", str(l_port),
            "-c", str(n_ctx),
        ])
        starting_llama_server()

    yield

    if llama_process is not None:
        llama_process.terminate()
        llama_process.wait(timeout=10)


app = FastAPI(lifespan=lifespan)

# Get the directory of this file
BASE_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Everything lives in memory for now so the sidebar list can too
ingested: list[IngestResult] = []


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ingest")
async def ingest_data(file: UploadFile):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")
    path_to_save = UPLOAD_DIR / file.filename
    with path_to_save.open("wb") as f:
        f.write(await file.read())

    result = process_and_store_data(str(path_to_save))
    if result.status == "success":
        ingested.append(result)
    return result


@app.get("/ingestions")
async def list_ingestions() -> list[IngestResult]:
    return ingested


@app.post("/query")
async def process_query(req: QueryRequest) -> QueryResponse:
    query = req.text.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # the reranker is sync and blocks for a while, keep it off the event loop
    ranked = await asyncio.to_thread(query_handler, query)
    return await answer_query(query, ranked, top_k=req.top_k)
