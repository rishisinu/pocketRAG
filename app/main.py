from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.core.ingestion import process_and_store_data
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
