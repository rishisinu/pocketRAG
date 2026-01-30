from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

app = FastAPI()

# Get the directory of this file
BASE_DIR = Path(__file__).parent

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.post("/ingest")
def ingest_data(data: dict):
    path = data.get("file_path")
    result = prcoess_and_store_data(path)
    return {"status": "success", "details": result}