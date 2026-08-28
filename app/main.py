from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.core.ingestion import process_and_store_data
from app.core.models import QueryRequest
from app.core.query_crossencoder import query_handler

import os
import subprocess
import time
import urllib.request
import urllib.error
from contextlib import asynccontextmanager

SERVER_BIN = os.environ["LLAMA_SERVER_BIN"]
MODEL_PATH = os.environ["LLAMA_MODEL_PATH"]
l_port = 8080
llama_health = f"http://localhost:{l_port}/health"
llama_process: subprocess.Popen[bytes] | None = None


def starting_llama_server(timeout: float = 60, interval: float = 0.5):
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        try:
            urllib.request.urlopen(llama_health, timeout=1)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(interval)
    raise RuntimeError("llama cpp didnt boot")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_process
    llama_process = subprocess.Popen([
        SERVER_BIN,
        "-m", MODEL_PATH,
        "--port", str(l_port),
    ])
    starting_llama_server()

    yield

    llama_process.terminate()
    llama_process.wait(timeout=10)
app = FastAPI(lifespan=lifespan)


app = FastAPI()









# Get the directory of this file
BASE_DIR = Path(__file__).parent

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
@app.post("/ingest")
async def ingest_data(file: UploadFile):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")
    path_to_save = UPLOAD_DIR / file.filename
    with path_to_save.open("wb") as f:
        f.write(await file.read())

    result = process_and_store_data(str(path_to_save))
    return result

@app.post("/query")
async def process_query(req: QueryRequest):
    query = req.text
    resp = query_handler(query)
